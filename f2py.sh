#!/bin/bash
F2PY=f2py
FORT=gfortran
BASEDIR=$(dirname "$0")

#For VR part
FILES='find_domain_py.f90  get_flux_py.f90  get_ptcl_py.f90  jsamr2cell_py.f90  jsamr2cell_totnum_py.f90  js_gasmap_py.f90  js_getpt_ft.f90'

cd $BASEDIR/vr/fortran
for f in $FILES
do
    bn=$(basename "$f" .f90)
    $F2PY -m $bn --fcompiler=$FORT --f90flags='-fopenmp' -lgomp -c $f
done

