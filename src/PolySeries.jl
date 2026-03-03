# ==============================================================================
# This file is part of the PolySeries.jl Julia package.
#
# Author: Jinyu Wan
# Email: wan@frib.msu.edu
# Version: 1.0
# Created Date: 11-01-2023
# Modified Date: 11-13-2023


module PolySeries
using StaticArrays
include("mathfunc.jl")
include("polymap.jl")
include("ctps.jl")
export CTPS, PSDesc, pow, cst, element, findindex, assign!, reassign!
export add!, addto!, sub!, subfrom!, scale!, scaleadd!, copy!, zero!, mul!
export set_descriptor!, get_descriptor, clear_descriptor!
export PSWorkspace, borrow!, release!
export sin!, cos!, exp!, log!, sqrt!, sinh!, cosh!
export pow!
include("macro.jl")
export @tpsa

end