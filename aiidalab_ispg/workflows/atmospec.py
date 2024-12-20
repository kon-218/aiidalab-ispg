"""Base work chain to run an ORCA calculation"""
from aiida.engine import (
    ExitCode,
    ToContext,
    WorkChain,
    append_,
    if_,
    process_handler,
    run,
)
from aiida.orm import (
    Bool,
    Dict,
    Float,
    Int,
    List,
    SinglefileData,
    StructureData,
    TrajectoryData,
    to_aiida_type,
    Str
)
from aiida.plugins import CalculationFactory, DataFactory, WorkflowFactory

# from aiida_shell import launch_shell_job
import tempfile
import subprocess
import os
from .harmonic_wigner import generate_wigner_structures
from .optimization import RobustOptimizationWorkChain
#from ..app import repre_sample_2D
from .utils import (
    ConcatInputsToList,
    add_orca_wf_guess,
    extract_trajectory_arrays,
    pick_structure_from_trajectory,
    structures_to_trajectory,
)

Code = DataFactory("core.code.installed")
OrcaCalculation = CalculationFactory("orca.orca")
OrcaBaseWorkChain = WorkflowFactory("orca.base")


class OrcaExcitationWorkChain(OrcaBaseWorkChain):
    """A simple shim for UV/vis excitation in ORCA."""

    def _build_process_label(self) -> str:
        return "Excitation workflow"

    @classmethod
    def define(cls, spec):
        super().define(spec)
        spec.output(
            "excitations",
            valid_type=Dict,
            required=True,
            help="Excitation energies and oscillator strengths from a single-point excitations",
        )

    def extract_transitions_from_orca_output(self, orca_output_params):
        return {
            "oscillator_strengths": orca_output_params["etoscs"],
            # Orca returns excited state energies in cm^-1
            # Perhaps we should do the conversion here,
            # to make this less ORCA specific.
            "excitation_energies_cm": orca_output_params["etenergies"],
        }

    @process_handler(exit_codes=ExitCode(0), priority=600)
    def add_excitation_output(self, calculation):
        """Extract excitation energies and osc. strengths into a separate output node"""
        transitions = self.extract_transitions_from_orca_output(
            calculation.outputs.output_parameters
        )
        self.out("excitations", Dict(transitions).store())


class RepSampleWorkChain(WorkChain):
    """WorkChain for running representative sampling using repre_sample_2D."""

    @classmethod
    def define(cls, spec):
        super().define(spec)
        # Inputs
        spec.input('excitation_data', valid_type=List,
                   help='List of tuples containing (energy, oscillator_strength)')
        spec.input('n_samples', valid_type=Int,
                   help='Total number of geometries')
        spec.input('n_states', valid_type=Int,
                   help='Number of excited states per geometry')
        spec.input('sample_size', valid_type=Int,
                   help='Number of geometries to select')
        spec.input('cycles', valid_type=Int, default=lambda: Int(2000))
        spec.input('jobs', valid_type=Int, default=lambda: Int(16))
        spec.input('total_jobs', valid_type=Int, default=lambda: Int(32))
        spec.input('weight_by_significance', valid_type=Bool, default=lambda: Bool(True))
        spec.input('pdf_comparison', valid_type=Str, default=lambda: Str('KLdiv'))

        # Outputs
        spec.output('selected_indices', valid_type=List)

        # Outline
        spec.outline(
            cls.setup_calculation,
            cls.run_representative_sampling,
            cls.process_results,
        )

    def setup_calculation(self):
        """Prepare input file for repre_sample_2D."""
        self.report("Setting up representative sampling calculation")
        # Create temporary input file
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            for energy, osc in self.inputs.excitation_data:
                f.write(f"{energy:.6f}\n")
                f.write(f"{osc:.6f}\n")
            self.ctx.input_file = f.name

    def run_representative_sampling(self):
        """
        Runs the representative sampling calculation using repre_sample_2D.py
        from the app directory. This method constructs and executes the sampling
        command with appropriate parameters.
        """
        try:
            # **Use the absolute path to the script here:**
            script_path = "/apps/aiidalab-ispg/aiidalab_ispg/app/repre_sample_2D.py"  # **Replace with the actual path**
        except FileNotFoundError as e:
            self.report(f"Error finding repre_sample_2D.py script: {str(e)}")
            return self.exit_codes.ERROR_MISSING_SCRIPT

        command = [
            "repre_sample_2D.py",
            "-n", str(self.inputs.n_samples.value),
            "-N", str(self.inputs.n_states.value),
            "-S", str(self.inputs.sample_size.value),
            "-c", str(self.inputs.cycles.value),
            "-j", str(self.inputs.jobs.value),
            "-J", str(self.inputs.total_jobs.value)
        ]

        if self.inputs.weight_by_significance.value:
            command.append("-w")
        command.extend([
            "--pdfcomp", self.inputs.pdf_comparison.value,
            self.ctx.input_file
        ])

        # Run the command and capture output
        result = subprocess.run(command, capture_output=True, text=True)
        self.ctx.output = result.stdout

        # Clean up input file
        os.unlink(self.ctx.input_file)

    def process_results(self):
        """Process output to get selected geometry indices."""
        # Parse output to get selected indices
        selected_indices = []
        for line in self.ctx.output.split('\n'):
            if line.strip().isdigit():
                # Convert from 1-based to 0-based indexing
                selected_indices.append(int(line.strip()) - 1)

        if not selected_indices:
            return self.exit_codes.ERROR_NO_SELECTED_INDICES

        self.out('selected_indices', List(selected_indices).store())


class OrcaWignerSpectrumWorkChain(WorkChain):
    """Top level workchain for Nuclear Ensemble Approach UV/vis
    spectrum for a single conformer"""

    def _build_process_label(self):
        return "NEA spectrum workflow"

    @classmethod
    def define(cls, spec):
        super().define(spec)
        spec.expose_inputs(
            RobustOptimizationWorkChain,
            namespace="opt",
            exclude=["orca.structure", "orca.code"],
        )
        spec.expose_inputs(
            OrcaExcitationWorkChain,
            namespace="exc",
            exclude=["orca.structure", "orca.code"],
        )
        spec.input("structure", valid_type=(StructureData, TrajectoryData))
        spec.input("code", valid_type=Code)
        # Whether to perform geometry optimization
        spec.input(
            "optimize",
            valid_type=Bool,
            default=lambda: Bool(True),
            serializer=to_aiida_type,
        )
        spec.input(
            "rep_sample",
            valid_type=Bool,
            default=lambda: Bool(True),
            serializer=to_aiida_type,
        )
        # Number of Wigner geometries (computed only when optimize==True)
        spec.input(
            "nwigner", valid_type=Int, default=lambda: Int(1), serializer=to_aiida_type
        )
        spec.input(
            "wigner_low_freq_thr",
            valid_type=Float,
            default=lambda: Float(10),
            serializer=to_aiida_type,
        )
        spec.output(
            "franck_condon_excitations",
            valid_type=Dict,
            required=True,
            help="Output parameters from a single-point excitations",
        )
        spec.expose_outputs(
            RobustOptimizationWorkChain,
            namespace="opt",
            include=["output_parameters", "relaxed_structure"],
            namespace_options={"required": False},
        )
        spec.output(
            "wigner_excitations",
            valid_type=List,
            required=False,
            help="Output parameters from all Wigner excited state calculation",
        )
        spec.output(
            "selected_representative_indices",
            valid_type=List,
            required=False,
            help="Indices of geometries selected by representative sampling."
        )
        spec.outline(
            if_(cls.should_optimize)(
                cls.optimize,
                cls.inspect_optimization,
            ),
            cls.excite,
            cls.inspect_excitation,
            if_(cls.should_run_wigner)(
                cls.wigner_sampling,
                cls.wigner_excite,
                cls.inspect_wigner_excitation,
                if_(cls.should_run_repsample)(
                    cls.repsample,
                    cls.inspect_repsample,
                ),
            ),
        )

        spec.exit_code(
            401,
            "ERROR_OPTIMIZATION_FAILED",
            "optimization encountered unspecified error",
        )
        spec.exit_code(
            402, "ERROR_EXCITATION_FAILED", "excited state calculation failed"
        )
        spec.exit_code(
            403,
            "ERROR_REPRESENTATIVE_SAMPLING_FAILED",
            "Representative sampling failed to produce output."
        )

    def excite(self):
        """Calculate excited states for a single geometry"""
        inputs = self.exposed_inputs(
            OrcaExcitationWorkChain, namespace="exc", agglomerate=False
        )
        inputs.orca.code = self.inputs.code
        if self.inputs.optimize:
            self.report("Calculating spectrum for optimized geometry")
            inputs.orca.structure = self.ctx.calc_opt.outputs.relaxed_structure
            # Pass in converged SCF wavefunction
            with self.ctx.calc_opt.outputs.retrieved.base.repository.open(
                "aiida.gbw", "rb"
            ) as handler:
                gbw_file = SinglefileData(handler)
                inputs.orca.file = {"gbw": gbw_file}
            inputs.orca.parameters = add_orca_wf_guess(inputs.orca.parameters)
        else:
            self.report("Calculating spectrum for input geometry")
            inputs.orca.structure = self.inputs.structure

        calc_exc = self.submit(OrcaExcitationWorkChain, **inputs)
        calc_exc.label = "franck-condon-excitation"
        return ToContext(calc_exc=calc_exc)

    def repsample(self):
        """Run representative sampling on excitation data using repre_sample_2D approach."""
        self.report("Starting representative sampling")

        # Gather excitation data from all Wigner calculations
        excitation_data = []

        for calc in self.ctx.wigner_calcs: # iterate over individual calculations
            if not calc.is_finished_ok:
                self.report(f"Skipping failed Wigner calculation {calc.pk}")
                continue

            # Extract the excitation energies and oscillator strengths
            data = calc.outputs.excitations.get_dict()
            energies = data["excitation_energies_cm"]

            # Convert from cm^-1 to eV (1 cm^-1 = 1.23984193 × 10^-4 eV)
            energies_ev = [e * 1.23984193e-4 for e in energies]
            oscs = data["oscillator_strengths"]

            # Add this geometry's data
            excitation_data.append((energies_ev, oscs))

        if not excitation_data:
            self.report("No valid excitation data available for representative sampling")
            return self.exit_codes.ERROR_REPRESENTATIVE_SAMPLING_FAILED

        # Format data for representative sampling
        input_data = []
        for energies, oscs in excitation_data:
            for energy, osc in zip(energies, oscs):
                input_data.append((energy, osc))

        # Create inputs for RepSampleWorkChain
        inputs = {
            "excitation_data": List(input_data).store(),
            "n_samples": self.inputs.nwigner,
            "n_states": Int(len(excitation_data)),  # Number of excited states
            "sample_size": Int(20),  # Number of geometries to select
            "cycles": Int(2000),
            "jobs": Int(16),
            "total_jobs": Int(32),
            "weight_by_significance": Bool(True),
            "pdf_comparison": Str("KLdiv")
        }

        # Submit the representative sampling calculation
        repsample_calc = self.submit(RepSampleWorkChain, **inputs)

        return ToContext(repsample_calc=repsample_calc)

    def inspect_repsample(self):
        """Check the results of representative sampling."""
        if not self.ctx.repsample_calc.is_finished_ok:
            self.report("Representative sampling failed")
            return self.exit_codes.ERROR_REPRESENTATIVE_SAMPLING_FAILED

        # Get selected geometry indices and store them
        selected_indices = self.ctx.repsample_calc.outputs.selected_indices
        self.report(f"Representative sampling selected geometries: {selected_indices}")
        self.out("selected_representative_indices", List(selected_indices).store())

    def wigner_sampling(self):
        self.report(f"Generating {self.inputs.nwigner.value} Wigner geometries")
        n_low_freq_vibs = 0
        for freq in self.ctx.calc_opt.outputs.output_parameters["vibfreqs"]:
            if freq < self.inputs.wigner_low_freq_thr:
                n_low_freq_vibs += 1
        if n_low_freq_vibs > 0:
            self.report(
                f"Ignoring {n_low_freq_vibs} vibrations below {self.inputs.wigner_low_freq_thr.value} cm^-1"
            )

        self.ctx.wigner_structures = generate_wigner_structures(
            self.ctx.calc_opt.outputs.relaxed_structure,
            self.ctx.calc_opt.outputs.output_parameters,
            self.inputs.nwigner,
            self.inputs.wigner_low_freq_thr,
        )

    def wigner_excite(self):
        inputs = self.exposed_inputs(
            OrcaExcitationWorkChain, namespace="exc", agglomerate=False
        )
        inputs.orca.code = self.inputs.code

        # Pass in SCF wavefunction from minimum geometry
        with self.ctx.calc_opt.outputs.retrieved.base.repository.open(
            "aiida.gbw", "rb"
        ) as handler:
            gbw_file = SinglefileData(handler)
            inputs.orca.file = {"gbw": gbw_file}
        inputs.orca.parameters = add_orca_wf_guess(inputs.orca.parameters)

        for i in self.ctx.wigner_structures.get_stepids():
            inputs.orca.structure = pick_structure_from_trajectory(
                self.ctx.wigner_structures, Int(i)
            )
            calc = self.submit(OrcaExcitationWorkChain, **inputs)
            calc.label = f"wigner-excitation-{i}"
            self.to_context(wigner_calcs=append_(calc))

    def optimize(self):
        """Optimize geometry"""
        inputs = self.exposed_inputs(
            RobustOptimizationWorkChain, namespace="opt", agglomerate=False
        )
        inputs.orca.structure = self.inputs.structure
        inputs.orca.code = self.inputs.code
        calc_opt = self.submit(RobustOptimizationWorkChain, **inputs)
        calc_opt.label = "optimization"
        return ToContext(calc_opt=calc_opt)

    def inspect_optimization(self):
        """Check whether optimization succeeded"""
        if not self.ctx.calc_opt.is_finished_ok:
            self.report("Optimization failed :-(")
            return self.exit_codes.ERROR_OPTIMIZATION_FAILED
        self.out_many(
            self.exposed_outputs(
                self.ctx.calc_opt,
                RobustOptimizationWorkChain,
                namespace="opt",
                agglomerate=False,
            )
        )

    def inspect_excitation(self):
        """Check whether excitation succeeded"""
        calc = self.ctx.calc_exc
        if not calc.is_finished_ok:
            self.report("Single point excitation failed :-(")
            return self.exit_codes.ERROR_EXCITATION_FAILED
        self.out("franck_condon_excitations", calc.outputs.excitations)

    def inspect_wigner_excitation(self):
        """Check whether all wigner excitations succeeded"""
        for calc in self.ctx.wigner_calcs:
            if not calc.is_finished_ok:
                self.report("Wigner excitation failed :-(")
                return self.exit_codes.ERROR_EXCITATION_FAILED
        all_wigner_data = [
            wc.outputs.excitations.get_dict() for wc in self.ctx.wigner_calcs
        ]
        self.report("Wigner excitation sucessfull")
        self.out("wigner_excitations", List(all_wigner_data).store())

    def should_optimize(self):
        return self.inputs.optimize.value

    def should_run_wigner(self):
        return self.should_optimize() and self.inputs.nwigner > 0

    def should_run_repsample(self):
         return self.inputs.rep_sample and self.inputs.nwigner > 0

class AtmospecWorkChain(WorkChain):
    """The top-level ATMOSPEC workchain"""

    def _build_process_label(self):
        return "ATMOSPEC workflow"

    @classmethod
    def define(cls, spec):
        super().define(spec)
        spec.expose_inputs(OrcaWignerSpectrumWorkChain, exclude=["structure"])
        spec.input("structure", valid_type=TrajectoryData)
        spec.output(
            "spectrum_data",
            valid_type=List,
            required=True,
            help="All data necessary to construct spectrum in SpectrumWidget",
        )
        spec.output(
            "relaxed_structures",
            valid_type=TrajectoryData,
            required=False,
            help="Minimized structures of all conformers",
        )
        spec.outline(
            cls.launch,
            cls.collect,
        )

    def launch(self):
        inputs = self.exposed_inputs(OrcaWignerSpectrumWorkChain, agglomerate=False)
        self.report(
            f"Launching ATMOSPEC for {len(self.inputs.structure.get_stepids())} conformers"
        )
        for conf_id in self.inputs.structure.get_stepids():
            inputs.structure = self.inputs.structure.get_step_structure(conf_id)
            workflow = self.submit(OrcaWignerSpectrumWorkChain, **inputs)
            workflow.label = f"atmospec-conf-{conf_id}"
            self.to_context(confs=append_(workflow))

    def collect(self):
        for wc in self.ctx.confs:
            if not wc.is_finished_ok:
                return ExitCode(wc.exit_status, wc.exit_message)

        conf_outputs = [wc.outputs for wc in self.ctx.confs]

        # Combine all spectra data
        if self.inputs.optimize and self.inputs.nwigner > 0:
            data = {
                str(i): outputs.wigner_excitations
                for i, outputs in enumerate(conf_outputs)
            }
        else:
            data = {
                str(i): [outputs.franck_condon_excitations.get_dict()]
                for i, outputs in enumerate(conf_outputs)
            }

        all_results = run(ConcatInputsToList, ns=data)
        self.out("spectrum_data", all_results["output"])

        # Combine all optimized geometries into single TrajectoryData
        if self.inputs.optimize:
            relaxed_structures = {}
            orca_output_params = {}
            for i, outputs in enumerate(conf_outputs):
                relaxed_structures[f"struct_{i}"] = outputs.opt.relaxed_structure
                orca_output_params[f"params_{i}"] = outputs.opt.output_parameters

            # For multiple conformers, we're appending relative energies and Boltzmann weights
            array_data = None
            if len(self.ctx.confs) > 1:
                array_data = extract_trajectory_arrays(**orca_output_params)

            trajectory = structures_to_trajectory(
                arrays=array_data, **relaxed_structures
            )
            self.out("relaxed_structures", trajectory)
