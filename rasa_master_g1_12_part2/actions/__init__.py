"""Actions package for merged grades.

This package imports actions from separate modules to avoid filename overwrites.
If you have duplicate action `name()` values across modules, rename one of them.
"""

from .grade_1_3_actions import *  # noqa
from .grade_4_6_actions import *  # noqa
from .grade_7_9_part1_actions import *  # noqa
from .grade_7_9_part2_actions import *

# NOTE: grade_7_9_part3_actions.py contains duplicate action names with earlier modules,
# so it is NOT imported by default. If you want to use it, rename conflicting action `name()` values,
# then add: from .grade_7_9_part3_actions import *

# NOTE: grade_10_12_part2_extra_actions.py saved as reference (not auto-imported to avoid action-name conflicts).
