# -*- coding: utf-8 -*-


class KPath:
    def __init__(
        self,
        knames=None,
        special_kpoints=None,
        ngrids=None,
        has_time_reversal=True,
    ):
        self.knames = knames
        self.special_kpoints = special_kpoints
        self.ngrids = ngrids
        self.has_time_reversal = has_time_reversal
