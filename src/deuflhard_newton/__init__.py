from pathlib import Path

from deuflhard_newton.nleq import nleq_err, nleq_res, ProblemType

package_root_abspath = Path(__file__).parent
project_root_abspath = package_root_abspath.parent.parent

__all__ = ['nleq_err', 'nleq_res', 'ProblemType']
