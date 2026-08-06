"""Load FastJet + fjcontrib (SoftDrop, Nsubjettiness, EnergyCorrelator) via cppyy."""
from __future__ import annotations

import sys
from functools import lru_cache


@lru_cache(maxsize=1)
def load_fjcontrib():
    import cppyy

    prefix = sys.prefix
    cppyy.add_include_path(f"{prefix}/include")
    cppyy.add_library_path(f"{prefix}/lib")
    for lib in ("fastjet", "fastjettools", "fastjetcontribfragile"):
        cppyy.load_library(lib)
    cppyy.include("vector")
    cppyy.include("fastjet/PseudoJet.hh")
    cppyy.include("fastjet/JetDefinition.hh")
    cppyy.include("fastjet/ClusterSequence.hh")
    cppyy.include("fastjet/contrib/SoftDrop.hh")
    cppyy.include("fastjet/contrib/Nsubjettiness.hh")
    cppyy.include("fastjet/contrib/EnergyCorrelator.hh")
    from cppyy.gbl import fastjet as fj
    from cppyy.gbl import std

    return fj, std
