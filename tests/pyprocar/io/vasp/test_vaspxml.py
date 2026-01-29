from typing import NamedTuple

import numpy as np
import pytest

from pyprocar.io.vasp.vasprun import VaspXML

PARAMETER_ELEMENT = """<parameters>
  <separator name="general" >
   <i type="string" name="SYSTEM">Default</i>
   <i type="logical" name="LCOMPAT"> F  </i>
  </separator>
  <separator name="electronic" >
   <i type="string" name="PREC">accura</i>
   <i name="ENMAX">    600.00000000</i>
   <i name="ENAUG">    605.39200000</i>
   <i name="EDIFF">      0.00000001</i>
   <i type="int" name="IALGO">    38</i>
   <i type="int" name="IWAVPR">    10</i>
   <i type="int" name="NBANDS">    20</i>
   <i type="int" name="NBANDSLOW">    -1</i>
   <i type="int" name="NBANDSHIGH">    -1</i>
   <i name="NELECT">     33.00000000</i>
   <i type="int" name="TURBO">     0</i>
   <i type="int" name="IRESTART">     0</i>
   <i type="int" name="NREBOOT">     0</i>
   <i type="int" name="NMIN">     0</i>
   <i name="EREF">      0.00000000</i>
   <separator name="electronic smearing" >
    <i type="int" name="ISMEAR">     2</i>
    <i name="SIGMA">      0.20000000</i>
    <i name="KSPACING">      0.50000000</i>
    <i type="logical" name="KGAMMA"> T  </i>
    <i type="logical" name="KBLOWUP"> T  </i>
   </separator>
   <separator name="electronic projectors" >
    <i type="logical" name="LREAL"> T  </i>
    <v name="ROPT">     -0.00025000     -0.00025000     -0.00025000</v>
    <i type="int" name="LMAXPAW">  -100</i>
    <i type="int" name="LMAXMIX">     4</i>
    <i type="logical" name="NLSPLINE"> F  </i>
   </separator>
   <separator name="electronic startup" >
    <i type="int" name="ISTART">     0</i>
    <i type="int" name="ICHARG">    11</i>
    <i type="int" name="INIWAV">     1</i>
   </separator>
   <separator name="electronic spin" >
    <i type="int" name="ISPIN">     1</i>
    <i type="logical" name="LNONCOLLINEAR"> F  </i>
    <v name="MAGMOM">      1.00000000      1.00000000      1.00000000      1.00000000      1.00000000</v>
    <i name="NUPDOWN">     -1.00000000</i>
    <i type="logical" name="LSORBIT"> F  </i>
    <v name="SAXIS">      0.00000000      0.00000000      1.00000000</v>
    <i type="logical" name="LSPIRAL"> F  </i>
    <v name="QSPIRAL">      0.00000000      0.00000000      0.00000000</v>
    <i type="logical" name="LZEROZ"> F  </i>
   </separator>
   <separator name="electronic exchange-correlation" >
    <i type="logical" name="LASPH"> T  </i>
   </separator>
   <separator name="electronic convergence" >
    <i type="int" name="NELM">   100</i>
    <i type="int" name="NELMDL">    -5</i>
    <i type="int" name="NELMIN">     8</i>
    <i name="ENINI">    600.00000000</i>
    <separator name="electronic convergence detail" >
     <i type="logical" name="LDIAG"> T  </i>
     <i type="logical" name="LSUBROT"> F  </i>
     <i name="WEIMIN">      0.00000000</i>
     <i name="EBREAK">      0.00000000</i>
     <i name="DEPER">      0.30000000</i>
     <i type="int" name="NRMM">     4</i>
     <i name="TIME">      0.40000000</i>
    </separator>
   </separator>
   <separator name="electronic mixer" >
    <i name="AMIX">      0.40000000</i>
    <i name="BMIX">      1.00000000</i>
    <i name="AMIN">      0.10000000</i>
    <i name="AMIX_MAG">      1.60000000</i>
    <i name="BMIX_MAG">      1.00000000</i>
    <separator name="electronic mixer details" >
     <i type="int" name="IMIX">     4</i>
     <i type="logical" name="MIXFIRST"> F  </i>
     <i type="int" name="MAXMIX">   -45</i>
     <i name="WC">    100.00000000</i>
     <i type="int" name="INIMIX">     1</i>
     <i type="int" name="MIXPRE">     1</i>
     <i type="int" name="MREMOVE">     5</i>
    </separator>
   </separator>
   <separator name="electronic dipolcorrection" >
    <i type="logical" name="LDIPOL"> F  </i>
    <i type="logical" name="LMONO"> F  </i>
    <i type="int" name="IDIPOL">     0</i>
    <i name="EPSILON">      1.00000000</i>
    <v name="DIPOL">   -100.00000000   -100.00000000   -100.00000000</v>
    <i name="EFIELD">      0.00000000</i>
    <i type="logical" name="LVACPOTAV"> F  </i>
   </separator>
  </separator>
  <separator name="grids" >
   <i type="int" name="NGX">    32</i>
   <i type="int" name="NGY">    32</i>
   <i type="int" name="NGZ">    32</i>
   <i type="int" name="NGXF">    64</i>
   <i type="int" name="NGYF">    64</i>
   <i type="int" name="NGZF">    64</i>
   <i type="logical" name="ADDGRID"> T  </i>
  </separator>
  <separator name="ionic" >
   <i type="int" name="NSW">     0</i>
   <i type="int" name="IBRION">    -1</i>
   <i type="int" name="MDALGO">     0</i>
   <i type="int" name="ISIF">     2</i>
   <i name="PSTRESS">      0.00000000</i>
   <i name="EDIFFG">      0.00000010</i>
   <i type="int" name="NFREE">     0</i>
   <i name="POTIM">      0.50000000</i>
   <i name="SMASS">     -3.00000000</i>
   <i name="SCALEE">      1.00000000</i>
  </separator>
  <separator name="ionic md" >
   <i name="TEBEG">      0.00010000</i>
   <i name="TEEND">      0.00010000</i>
   <i type="int" name="NBLOCK">     1</i>
   <i type="int" name="KBLOCK">     1</i>
   <i type="int" name="NPACO">   256</i>
   <i name="APACO">     10.00000000</i>
  </separator>
  <separator name="symmetry" >
   <i type="int" name="ISYM">     2</i>
   <i name="SYMPREC">      0.00001000</i>
  </separator>
  <separator name="dos" >
   <i type="int" name="LORBIT">    11</i>
   <v name="RWIGS">     -1.00000000     -1.00000000     -1.00000000</v>
   <i type="int" name="NEDOS">   301</i>
   <i name="EMIN">     10.00000000</i>
   <i name="EMAX">    -10.00000000</i>
   <i name="EFERMI">      0.00000000</i>
  </separator>
  <separator name="writing" >
   <i type="int" name="NWRITE">     2</i>
   <i type="logical" name="LWAVE"> F  </i>
   <i type="logical" name="LDOWNSAMPLE"> F  </i>
   <i type="logical" name="LCHARG"> T  </i>
   <i type="logical" name="LPARD"> F  </i>
   <i type="logical" name="LVTOT"> F  </i>
   <i type="logical" name="LVHAR"> F  </i>
   <i type="logical" name="LELF"> F  </i>
   <i type="logical" name="LOPTICS"> F  </i>
   <v name="STM">      0.00000000      0.00000000      0.00000000      0.00000000      0.00000000      0.00000000      0.00000000</v>
  </separator>
  <separator name="performance" >
   <i type="int" name="NPAR">    10</i>
   <i type="int" name="NSIM">     4</i>
   <i type="int" name="NBLK">    -1</i>
   <i type="logical" name="LPLANE"> T  </i>
   <i type="logical" name="LSCALAPACK"> T  </i>
   <i type="logical" name="LSCAAWARE"> F  </i>
   <i type="logical" name="LSCALU"> F  </i>
   <i type="logical" name="LASYNC"> F  </i>
   <i type="logical" name="LORBITALREAL"> F  </i>
  </separator>
  <separator name="miscellaneous" >
   <i type="int" name="IDIOT">     3</i>
   <i type="int" name="PHON_NSTRUCT">    -1</i>
   <i type="logical" name="LMUSIC"> F  </i>
   <v name="POMASS">     87.62000000     50.94100000     16.00000000</v>
   <v name="DARWINR">      0.00000000      0.00000000      0.00000000</v>
   <v name="DARWINV">      1.00000000      1.00000000      1.00000000</v>
   <i type="logical" name="LCORR"> T  </i>
  </separator>
  <i type="logical" name="GGA_COMPAT"> T  </i>
  <i type="logical" name="LBERRY"> F  </i>
  <i type="int" name="ICORELEVEL">     0</i>
  <i type="logical" name="LDAU"> T  </i>
  <v type="int" name="LDAUTYPE">     1</v>
  <v type="int" name="LDAUL">    -1</v>
  <v name="LDAUU">      0.00000000</v>
  <v name="LDAUJ">      0.00000000</v>
  <i type="int" name="LDAUPRINT">     2</i>
  <i type="int" name="I_CONSTRAINED_M">     0</i>
  <separator name="electronic exchange-correlation" >
   <i type="string" name="GGA    ">PE</i>
   <i type="string" name="XC_C">1</i>
   <i type="int" name="VOSKOWN">     0</i>
   <i type="logical" name="LHFCALC"> F  </i>
   <i type="string" name="PRECFOCK"></i>
   <i type="logical" name="LSYMGRAD"> F  </i>
   <i type="logical" name="LHFONE"> F  </i>
   <i type="logical" name="LRHFCALC"> F  </i>
   <i type="logical" name="LTHOMAS"> F  </i>
   <i type="logical" name="LMODELHF"> F  </i>
   <i type="logical" name="LFOCKACE"> F  </i>
   <i name="ENCUT4O">     -1.00000000</i>
   <i type="int" name="EXXOEP">     0</i>
   <i type="int" name="FOURORBIT">     0</i>
   <i name="AEXX">      0.00000000</i>
   <i name="HFALPHA">      0.00000000</i>
   <i name="MCALPHA">      0.00000000</i>
   <i name="ALDAX">      1.00000000</i>
   <i name="AGGAX">      1.00000000</i>
   <i name="AMGGAX">      1.00000000</i>
   <i name="ALDAC">      1.00000000</i>
   <i name="AGGAC">      1.00000000</i>
   <i name="AMGGAC">      1.00000000</i>
   <i type="int" name="NKREDX">     1</i>
   <i type="int" name="NKREDY">     1</i>
   <i type="int" name="NKREDZ">     1</i>
   <i type="logical" name="SHIFTRED"> F  </i>
   <i type="logical" name="ODDONLY"> F  </i>
   <i type="logical" name="EVENONLY"> F  </i>
   <i type="int" name="LMAXFOCK">     0</i>
   <i type="int" name="NMAXFOCKAE">     0</i>
   <i type="logical" name="LFOCKAEDFT"> F  </i>
   <i name="HFSCREEN">      0.00000000</i>
   <i name="HFSCREENC">      0.00000000</i>
   <i type="int" name="NBANDSGWLOW">     0</i>
  </separator>
  <separator name="vdW DFT" >
   <i type="logical" name="LUSE_VDW"> F  </i>
   <i type="int" name="IVDW_NL">    -1</i>
   <i type="logical" name="LSPIN_VDW"> F  </i>
   <i name="ZAB_VDW">     -0.84910000</i>
   <i name="GAMMA_VDW">      1.39626340</i>
   <i name="ALPHA_VDW">      0.00000000</i>
   <i name="PARAM1">      0.12340000</i>
   <i name="PARAM2">      1.00000000</i>
   <i name="BPARAM">      6.30000000</i>
   <i name="CPARAM">      0.00930000</i>
  </separator>
  <separator name="model GW" >
   <i type="int" name="MODEL_GW">     0</i>
   <i name="MODEL_EPS0">      6.82944215</i>
   <i name="MODEL_ALPHA">      1.00000000</i>
  </separator>
  <separator name="linear response parameters" >
   <i type="logical" name="LEPSILON"> F  </i>
   <i type="logical" name="LRPA"> F  </i>
   <i type="logical" name="LNABLA"> F  </i>
   <i type="logical" name="LVEL"> F  </i>
   <i name="CSHIFT">      0.10000000</i>
   <i name="OMEGAMAX">     -1.00000000</i>
   <i name="DEG_THRESHOLD">      0.00200000</i>
   <i name="RTIME">     -0.10000000</i>
   <i name="WPLASMAI">      0.00000000</i>
   <v name="DFIELD">      0.00000000      0.00000000      0.00000000</v>
   <v name="WPLASMA">      0.00000000      0.00000000      0.00000000      0.00000000      0.00000000      0.00000000      0.00000000      0.00000000      0.00000000</v>
  </separator>
  <separator name="orbital magnetization" >
   <i type="logical" name="NUCIND"> F  </i>
   <v name="MAGPOS">      0.00000000      0.00000000      0.00000000</v>
   <i type="logical" name="LNICSALL"> T  </i>
   <i type="logical" name="ORBITALMAG"> F  </i>
   <i type="logical" name="LMAGBLOCH"> F  </i>
   <i type="logical" name="LCHIMAG"> F  </i>
   <i type="logical" name="LGAUGE"> T  </i>
   <i type="int" name="MAGATOM">     0</i>
   <v name="MAGDIPOL">      0.00000000      0.00000000      0.00000000</v>
   <v name="AVECCONST">      0.00000000      0.00000000      0.00000000</v>
  </separator>
  <separator name="response functions" >
   <i type="logical" name="LALL_IN_ONE"> F  </i>
   <i type="int" name="IALL_IN_ONE">    -1</i>
   <i type="int" name="NBANDS_WAVE">    -1</i>
   <i type="logical" name="LFINITE_TEMPERATURE"> F  </i>
   <i type="logical" name="LADDER"> F  </i>
   <i type="logical" name="LRPAFORCE"> F  </i>
   <i type="logical" name="LFXC"> F  </i>
   <i type="logical" name="LHARTREE"> T  </i>
   <i type="int" name="IBSE">     0</i>
   <v type="int" name="KPOINT">    -1     0     0     0</v>
   <i type="logical" name="LTCTC"> F  </i>
   <i type="logical" name="LTCTE"> F  </i>
   <i type="logical" name="LTETE"> F  </i>
   <i type="logical" name="LTRIPLET"> F  </i>
   <i type="logical" name="LFXCEPS"> F  </i>
   <i type="logical" name="LFXHEG"> F  </i>
   <i type="int" name="NATURALO">     2</i>
   <i type="logical" name="LHOLEGF"> F  </i>
   <i type="logical" name="L2ORDER"> F  </i>
   <i type="logical" name="LDMP1"> F  </i>
   <i type="logical" name="LMP2LT"> F  </i>
   <i type="logical" name="LSMP2LT"> F  </i>
   <i type="logical" name="LGWLF"> F  </i>
   <i name="ENCUTGW">     -1.60000002</i>
   <i name="ENCUTGWSOFT">     -1.60000002</i>
   <i name="ENCUTLF">     -1.00000000</i>
   <i type="logical" name="ESF_SPLINES"> F  </i>
   <i name="ESF_CONV">      0.01000000</i>
   <i type="int" name="ESF_NINTER">    15</i>
   <i type="int" name="LMAXMP2">    -1</i>
   <i name="SCISSOR">      0.00000000</i>
   <i type="int" name="NOMEGA">     0</i>
   <i type="int" name="NOMEGAR">     0</i>
   <i type="int" name="NBANDSGW">    -1</i>
   <i type="int" name="NBANDSO">    -1</i>
   <i type="int" name="NBANDSV">    -1</i>
   <i type="int" name="NELMGW">     1</i>
   <i type="int" name="NELMHF">     1</i>
   <i type="int" name="DIM">     3</i>
   <i type="int" name="IESPILON">     4</i>
   <i type="int" name="ANTIRES">     0</i>
   <i name="OMEGAMAX">    -30.00000000</i>
   <i name="OMEGAMIN">    -30.00000000</i>
   <i name="OMEGATL">   -200.00000000</i>
   <i type="int" name="OMEGAGRID">     0</i>
   <i name="CSHIFT">     -0.10000000</i>
   <i type="logical" name="LSELFENERGY"> F  </i>
   <i type="logical" name="LSPECTRAL"> F  </i>
   <i type="logical" name="LSPECTRALGW"> F  </i>
   <i type="logical" name="LSINGLES"> F  </i>
   <i type="logical" name="LFERMIGW"> F  </i>
   <i type="logical" name="ODDONLYGW"> F  </i>
   <i type="logical" name="EVENONLYGW"> F  </i>
   <i type="int" name="NKREDLFX">     1</i>
   <i type="int" name="NKREDLFY">     1</i>
   <i type="int" name="NKREDLFZ">     1</i>
   <i type="int" name="MAXMEM">  2800</i>
   <i type="int" name="TELESCOPE">     0</i>
   <i type="int" name="NTAUPAR">    -1</i>
   <i type="int" name="NOMEGAPAR">    -1</i>
   <i name="DAMP_NEWTON">      0.80000001</i>
   <i name="LAMBDA">      1.00000000</i>
  </separator>
  <separator name="External order field" >
   <i name="OFIELD_KAPPA">      0.00000000</i>
   <v name="OFIELD_K">      0.00000000      0.00000000      0.00000000</v>
   <i name="OFIELD_Q6_NEAR">      0.00000000</i>
   <i name="OFIELD_Q6_FAR">      0.00000000</i>
   <i name="OFIELD_A">      0.00000000</i>
  </separator>
  <separator name="optional k-points parameters" >
   <i type="int" name="KPOINTS_OPT_MODE">     1</i>
   <i type="logical" name="LKPOINTS_OPT"> F  </i>
  </separator>
 </parameters>
"""


class TestVaspXMLParameters:
    def test_parse_general_parameters(self) -> None:
        vaspxml = VaspXML.from_str(PARAMETER_ELEMENT)
        assert vaspxml.general_parameters is not None
        assert isinstance(vaspxml.general_parameters, dict)
        assert vaspxml.general_parameters["SYSTEM"] == "Default"
        assert vaspxml.general_parameters["LCOMPAT"] is False

    def test_electronic_parameters(self) -> None:
        vaspxml = VaspXML.from_str(PARAMETER_ELEMENT)
        assert vaspxml.electronic_parameters is not None
        assert isinstance(vaspxml.electronic_parameters, dict)
        assert vaspxml.electronic_parameters["PREC"] == "accura"
        assert vaspxml.electronic_parameters["ENMAX"] == 600.00000000
        assert vaspxml.electronic_parameters["ENAUG"] == 605.39200000
        assert vaspxml.electronic_parameters["EDIFF"] == 0.00000001
        assert vaspxml.electronic_parameters["IALGO"] == 38
        assert vaspxml.electronic_parameters["IWAVPR"] == 10
        assert vaspxml.electronic_parameters["NBANDS"] == 20
        assert vaspxml.electronic_parameters["NBANDSLOW"] == -1
        assert vaspxml.electronic_parameters["NBANDSHIGH"] == -1
        assert vaspxml.electronic_parameters["NELECT"] == 33.00000000
        assert vaspxml.electronic_parameters["TURBO"] == 0
        assert vaspxml.electronic_parameters["IRESTART"] == 0
        assert vaspxml.electronic_parameters["NREBOOT"] == 0
        assert vaspxml.electronic_parameters["NMIN"] == 0
        assert vaspxml.electronic_parameters["EREF"] == 0.00000000

    def test_electronic_smearing_parameters(self) -> None:
        vaspxml = VaspXML.from_str(PARAMETER_ELEMENT)
        assert vaspxml.electronic_smearing_parameters is not None
        assert isinstance(vaspxml.electronic_smearing_parameters, dict)
        assert vaspxml.electronic_smearing_parameters["ISMEAR"] == 2
        assert vaspxml.electronic_smearing_parameters["SIGMA"] == 0.20000000
        assert vaspxml.electronic_smearing_parameters["KSPACING"] == 0.50000000
        assert vaspxml.electronic_smearing_parameters["KGAMMA"] is True
        assert vaspxml.electronic_smearing_parameters["KBLOWUP"] is True

    def test_electronic_projectors_parameters(self) -> None:
        vaspxml = VaspXML.from_str(PARAMETER_ELEMENT)
        assert vaspxml.electronic_projectors_parameters is not None
        assert isinstance(vaspxml.electronic_projectors_parameters, dict)
        assert vaspxml.electronic_projectors_parameters["LREAL"] is True
        assert np.allclose(
            vaspxml.electronic_projectors_parameters["ROPT"],
            np.array([-0.00025000, -0.00025000, -0.00025000]),
        )
        assert vaspxml.electronic_projectors_parameters["LMAXPAW"] == -100
        assert vaspxml.electronic_projectors_parameters["LMAXMIX"] == 4
        assert vaspxml.electronic_projectors_parameters["NLSPLINE"] is False

    def test_electronic_startup_parameters(self) -> None:
        vaspxml = VaspXML.from_str(PARAMETER_ELEMENT)
        assert vaspxml.electronic_startup_parameters is not None
        assert isinstance(vaspxml.electronic_startup_parameters, dict)
        assert vaspxml.electronic_startup_parameters["ISTART"] == 0
        assert vaspxml.electronic_startup_parameters["ICHARG"] == 11
        assert vaspxml.electronic_startup_parameters["INIWAV"] == 1

    def test_electronic_spin_parameters(self) -> None:
        vaspxml = VaspXML.from_str(PARAMETER_ELEMENT)
        assert vaspxml.electronic_spin_parameters is not None
        assert isinstance(vaspxml.electronic_spin_parameters, dict)
        assert vaspxml.electronic_spin_parameters["ISPIN"] == 1
        assert vaspxml.electronic_spin_parameters["LNONCOLLINEAR"] is False
        assert np.allclose(
            vaspxml.electronic_spin_parameters["MAGMOM"], np.array([1.0, 1.0, 1.0, 1.0, 1.0])
        )
        assert vaspxml.electronic_spin_parameters["NUPDOWN"] == -1.00000000
        assert vaspxml.electronic_spin_parameters["LSORBIT"] is False
        assert np.allclose(vaspxml.electronic_spin_parameters["SAXIS"], np.array([0.0, 0.0, 1.0]))
        assert vaspxml.electronic_spin_parameters["LSPIRAL"] is False
        assert np.allclose(vaspxml.electronic_spin_parameters["QSPIRAL"], np.array([0.0, 0.0, 0.0]))
        assert vaspxml.electronic_spin_parameters["LZEROZ"] is False

    def test_electronic_exchange_correlation_parameters(self) -> None:
        vaspxml = VaspXML.from_str(PARAMETER_ELEMENT)
        assert vaspxml.electronic_exchange_correlation_parameters is not None
        assert isinstance(vaspxml.electronic_exchange_correlation_parameters, dict)
        assert vaspxml.electronic_exchange_correlation_parameters["LASPH"] is True

    def test_electronic_convergence_parameters(self) -> None:
        vaspxml = VaspXML.from_str(PARAMETER_ELEMENT)
        assert vaspxml.electronic_convergence_parameters is not None
        assert isinstance(vaspxml.electronic_convergence_parameters, dict)
        assert vaspxml.electronic_convergence_parameters["NELM"] == 100
        assert vaspxml.electronic_convergence_parameters["NELMDL"] == -5
        assert vaspxml.electronic_convergence_parameters["NELMIN"] == 8
        assert vaspxml.electronic_convergence_parameters["ENINI"] == 600.00000000

    def test_electronic_convergence_detail_parameters(self) -> None:
        vaspxml = VaspXML.from_str(PARAMETER_ELEMENT)
        assert vaspxml.electronic_convergence_detail_parameters is not None
        assert isinstance(vaspxml.electronic_convergence_detail_parameters, dict)
        assert vaspxml.electronic_convergence_detail_parameters["LDIAG"] is True
        assert vaspxml.electronic_convergence_detail_parameters["LSUBROT"] is False
        assert vaspxml.electronic_convergence_detail_parameters["WEIMIN"] == 0.00000000
        assert vaspxml.electronic_convergence_detail_parameters["EBREAK"] == 0.00000000
        assert vaspxml.electronic_convergence_detail_parameters["DEPER"] == 0.30000000
        assert vaspxml.electronic_convergence_detail_parameters["NRMM"] == 4
        assert vaspxml.electronic_convergence_detail_parameters["TIME"] == 0.40000000

    def test_electronic_mixer_parameters(self) -> None:
        vaspxml = VaspXML.from_str(PARAMETER_ELEMENT)
        assert vaspxml.electronic_mixer_parameters is not None
        assert isinstance(vaspxml.electronic_mixer_parameters, dict)
        assert vaspxml.electronic_mixer_parameters["AMIX"] == 0.40000000
        assert vaspxml.electronic_mixer_parameters["BMIX"] == 1.00000000
        assert vaspxml.electronic_mixer_parameters["AMIN"] == 0.10000000
        assert vaspxml.electronic_mixer_parameters["AMIX_MAG"] == 1.60000000
        assert vaspxml.electronic_mixer_parameters["BMIX_MAG"] == 1.00000000

    def test_electronic_mixer_details_parameters(self) -> None:
        vaspxml = VaspXML.from_str(PARAMETER_ELEMENT)
        assert vaspxml.electronic_mixer_details_parameters is not None
        assert isinstance(vaspxml.electronic_mixer_details_parameters, dict)
        assert vaspxml.electronic_mixer_details_parameters["IMIX"] == 4
        assert vaspxml.electronic_mixer_details_parameters["MIXFIRST"] is False
        assert vaspxml.electronic_mixer_details_parameters["MAXMIX"] == -45
        assert vaspxml.electronic_mixer_details_parameters["WC"] == 100.00000000
        assert vaspxml.electronic_mixer_details_parameters["INIMIX"] == 1
        assert vaspxml.electronic_mixer_details_parameters["MIXPRE"] == 1
        assert vaspxml.electronic_mixer_details_parameters["MREMOVE"] == 5

    def test_electronic_dipolcorrection_parameters(self) -> None:
        vaspxml = VaspXML.from_str(PARAMETER_ELEMENT)
        assert vaspxml.electronic_dipolcorrection_parameters is not None
        assert isinstance(vaspxml.electronic_dipolcorrection_parameters, dict)
        assert vaspxml.electronic_dipolcorrection_parameters["LDIPOL"] is False
        assert vaspxml.electronic_dipolcorrection_parameters["LMONO"] is False
        assert vaspxml.electronic_dipolcorrection_parameters["IDIPOL"] == 0
        assert vaspxml.electronic_dipolcorrection_parameters["EPSILON"] == 1.00000000
        assert np.allclose(
            vaspxml.electronic_dipolcorrection_parameters["DIPOL"],
            np.array([-100.0, -100.0, -100.0]),
        )
        assert vaspxml.electronic_dipolcorrection_parameters["EFIELD"] == 0.00000000
        assert vaspxml.electronic_dipolcorrection_parameters["LVACPOTAV"] is False

    def test_grids_parameters(self) -> None:
        vaspxml = VaspXML.from_str(PARAMETER_ELEMENT)
        assert vaspxml.grids_parameters is not None
        assert isinstance(vaspxml.grids_parameters, dict)
        assert vaspxml.grids_parameters["NGX"] == 32
        assert vaspxml.grids_parameters["NGY"] == 32
        assert vaspxml.grids_parameters["NGZ"] == 32
        assert vaspxml.grids_parameters["NGXF"] == 64
        assert vaspxml.grids_parameters["NGYF"] == 64
        assert vaspxml.grids_parameters["NGZF"] == 64
        assert vaspxml.grids_parameters["ADDGRID"] is True

    def test_ionic_parameters(self) -> None:
        vaspxml = VaspXML.from_str(PARAMETER_ELEMENT)
        assert vaspxml.ionic_parameters is not None
        assert isinstance(vaspxml.ionic_parameters, dict)
        assert vaspxml.ionic_parameters["NSW"] == 0
        assert vaspxml.ionic_parameters["IBRION"] == -1
        assert vaspxml.ionic_parameters["MDALGO"] == 0
        assert vaspxml.ionic_parameters["ISIF"] == 2
        assert vaspxml.ionic_parameters["PSTRESS"] == 0.00000000
        assert vaspxml.ionic_parameters["EDIFFG"] == 0.00000010
        assert vaspxml.ionic_parameters["NFREE"] == 0
        assert vaspxml.ionic_parameters["POTIM"] == 0.50000000
        assert vaspxml.ionic_parameters["SMASS"] == -3.00000000
        assert vaspxml.ionic_parameters["SCALEE"] == 1.00000000

    def test_ionic_md_parameters(self) -> None:
        vaspxml = VaspXML.from_str(PARAMETER_ELEMENT)
        assert vaspxml.ionic_md_parameters is not None
        assert isinstance(vaspxml.ionic_md_parameters, dict)
        assert vaspxml.ionic_md_parameters["TEBEG"] == 0.00010000
        assert vaspxml.ionic_md_parameters["TEEND"] == 0.00010000
        assert vaspxml.ionic_md_parameters["NBLOCK"] == 1
        assert vaspxml.ionic_md_parameters["KBLOCK"] == 1
        assert vaspxml.ionic_md_parameters["NPACO"] == 256
        assert vaspxml.ionic_md_parameters["APACO"] == 10.00000000

    def test_symmetry_parameters(self) -> None:
        vaspxml = VaspXML.from_str(PARAMETER_ELEMENT)
        assert vaspxml.symmetry_parameters is not None
        assert isinstance(vaspxml.symmetry_parameters, dict)
        assert vaspxml.symmetry_parameters["ISYM"] == 2
        assert vaspxml.symmetry_parameters["SYMPREC"] == 0.00001000

    def test_dos_parameters(self) -> None:
        vaspxml = VaspXML.from_str(PARAMETER_ELEMENT)
        assert vaspxml.dos_parameters is not None
        assert isinstance(vaspxml.dos_parameters, dict)
        assert vaspxml.dos_parameters["LORBIT"] == 11
        assert np.allclose(vaspxml.dos_parameters["RWIGS"], np.array([-1.0, -1.0, -1.0]))
        assert vaspxml.dos_parameters["NEDOS"] == 301
        assert vaspxml.dos_parameters["EMIN"] == 10.00000000
        assert vaspxml.dos_parameters["EMAX"] == -10.00000000
        assert vaspxml.dos_parameters["EFERMI"] == 0.00000000

    def test_writing_parameters(self) -> None:
        vaspxml = VaspXML.from_str(PARAMETER_ELEMENT)
        assert vaspxml.writing_parameters is not None
        assert isinstance(vaspxml.writing_parameters, dict)
        assert vaspxml.writing_parameters["NWRITE"] == 2
        assert vaspxml.writing_parameters["LWAVE"] is False
        assert vaspxml.writing_parameters["LDOWNSAMPLE"] is False
        assert vaspxml.writing_parameters["LCHARG"] is True
        assert vaspxml.writing_parameters["LPARD"] is False
        assert vaspxml.writing_parameters["LVTOT"] is False
        assert vaspxml.writing_parameters["LVHAR"] is False
        assert vaspxml.writing_parameters["LELF"] is False
        assert vaspxml.writing_parameters["LOPTICS"] is False
        assert np.allclose(
            vaspxml.writing_parameters["STM"], np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        )

    def test_performance_parameters(self) -> None:
        vaspxml = VaspXML.from_str(PARAMETER_ELEMENT)
        assert vaspxml.performance_parameters is not None
        assert isinstance(vaspxml.performance_parameters, dict)
        assert vaspxml.performance_parameters["NPAR"] == 10
        assert vaspxml.performance_parameters["NSIM"] == 4
        assert vaspxml.performance_parameters["NBLK"] == -1
        assert vaspxml.performance_parameters["LPLANE"] is True
        assert vaspxml.performance_parameters["LSCALAPACK"] is True
        assert vaspxml.performance_parameters["LSCAAWARE"] is False
        assert vaspxml.performance_parameters["LSCALU"] is False
        assert vaspxml.performance_parameters["LASYNC"] is False
        assert vaspxml.performance_parameters["LORBITALREAL"] is False

    def test_miscellaneous_parameters(self) -> None:
        vaspxml = VaspXML.from_str(PARAMETER_ELEMENT)
        assert vaspxml.miscellaneous_parameters is not None
        assert isinstance(vaspxml.miscellaneous_parameters, dict)
        assert vaspxml.miscellaneous_parameters["IDIOT"] == 3
        assert vaspxml.miscellaneous_parameters["PHON_NSTRUCT"] == -1
        assert vaspxml.miscellaneous_parameters["LMUSIC"] is False
        assert np.allclose(
            vaspxml.miscellaneous_parameters["POMASS"],
            np.array([87.62000000, 50.94100000, 16.00000000]),
        )
        assert np.allclose(vaspxml.miscellaneous_parameters["DARWINR"], np.array([0.0, 0.0, 0.0]))
        assert np.allclose(vaspxml.miscellaneous_parameters["DARWINV"], np.array([1.0, 1.0, 1.0]))
        assert vaspxml.miscellaneous_parameters["LCORR"] is True

    def test_ldau_parameters(self) -> None:
        vaspxml = VaspXML.from_str(PARAMETER_ELEMENT)
        assert vaspxml.ldau_parameters is not None
        assert isinstance(vaspxml.ldau_parameters, dict)
        assert vaspxml.ldau_parameters["GGA_COMPAT"] is True
        assert vaspxml.ldau_parameters["LBERRY"] is False
        assert vaspxml.ldau_parameters["ICORELEVEL"] == 0
        assert vaspxml.ldau_parameters["LDAU"] is True
        assert np.allclose(vaspxml.ldau_parameters["LDAUTYPE"], np.array([1]))
        assert np.allclose(vaspxml.ldau_parameters["LDAUL"], np.array([-1]))
        assert np.allclose(vaspxml.ldau_parameters["LDAUU"], np.array([0.0]))
        assert np.allclose(vaspxml.ldau_parameters["LDAUJ"], np.array([0.0]))
        assert vaspxml.ldau_parameters["LDAUPRINT"] == 2
        assert vaspxml.ldau_parameters["I_CONSTRAINED_M"] == 0

    def test_exchange_correlation_parameters(self) -> None:
        vaspxml = VaspXML.from_str(PARAMETER_ELEMENT)
        assert vaspxml.exchange_correlation_parameters is not None
        assert isinstance(vaspxml.exchange_correlation_parameters, dict)
        assert vaspxml.exchange_correlation_parameters["GGA"] == "PE"
        assert vaspxml.exchange_correlation_parameters["XC_C"] == "1"
        assert vaspxml.exchange_correlation_parameters["VOSKOWN"] == 0
        assert vaspxml.exchange_correlation_parameters["LHFCALC"] is False
        assert vaspxml.exchange_correlation_parameters["PRECFOCK"] == ""
        assert vaspxml.exchange_correlation_parameters["LSYMGRAD"] is False
        assert vaspxml.exchange_correlation_parameters["LHFONE"] is False
        assert vaspxml.exchange_correlation_parameters["LRHFCALC"] is False
        assert vaspxml.exchange_correlation_parameters["LTHOMAS"] is False
        assert vaspxml.exchange_correlation_parameters["LMODELHF"] is False
        assert vaspxml.exchange_correlation_parameters["LFOCKACE"] is False
        assert vaspxml.exchange_correlation_parameters["ENCUT4O"] == -1.00000000
        assert vaspxml.exchange_correlation_parameters["EXXOEP"] == 0
        assert vaspxml.exchange_correlation_parameters["FOURORBIT"] == 0
        assert vaspxml.exchange_correlation_parameters["AEXX"] == 0.00000000
        assert vaspxml.exchange_correlation_parameters["HFALPHA"] == 0.00000000
        assert vaspxml.exchange_correlation_parameters["MCALPHA"] == 0.00000000
        assert vaspxml.exchange_correlation_parameters["ALDAX"] == 1.00000000
        assert vaspxml.exchange_correlation_parameters["AGGAX"] == 1.00000000
        assert vaspxml.exchange_correlation_parameters["AMGGAX"] == 1.00000000
        assert vaspxml.exchange_correlation_parameters["ALDAC"] == 1.00000000
        assert vaspxml.exchange_correlation_parameters["AGGAC"] == 1.00000000
        assert vaspxml.exchange_correlation_parameters["AMGGAC"] == 1.00000000
        assert vaspxml.exchange_correlation_parameters["NKREDX"] == 1
        assert vaspxml.exchange_correlation_parameters["NKREDY"] == 1
        assert vaspxml.exchange_correlation_parameters["NKREDZ"] == 1
        assert vaspxml.exchange_correlation_parameters["SHIFTRED"] is False
        assert vaspxml.exchange_correlation_parameters["ODDONLY"] is False
        assert vaspxml.exchange_correlation_parameters["EVENONLY"] is False
        assert vaspxml.exchange_correlation_parameters["LMAXFOCK"] == 0
        assert vaspxml.exchange_correlation_parameters["NMAXFOCKAE"] == 0
        assert vaspxml.exchange_correlation_parameters["LFOCKAEDFT"] is False
        assert vaspxml.exchange_correlation_parameters["HFSCREEN"] == 0.00000000
        assert vaspxml.exchange_correlation_parameters["HFSCREENC"] == 0.00000000
        assert vaspxml.exchange_correlation_parameters["NBANDSGWLOW"] == 0

    def test_vdw_dft_parameters(self) -> None:
        vaspxml = VaspXML.from_str(PARAMETER_ELEMENT)
        assert vaspxml.vdw_dft_parameters is not None
        assert isinstance(vaspxml.vdw_dft_parameters, dict)
        assert vaspxml.vdw_dft_parameters["LUSE_VDW"] is False
        assert vaspxml.vdw_dft_parameters["IVDW_NL"] == -1
        assert vaspxml.vdw_dft_parameters["LSPIN_VDW"] is False
        assert vaspxml.vdw_dft_parameters["ZAB_VDW"] == -0.84910000
        assert vaspxml.vdw_dft_parameters["GAMMA_VDW"] == 1.39626340
        assert vaspxml.vdw_dft_parameters["ALPHA_VDW"] == 0.00000000
        assert vaspxml.vdw_dft_parameters["PARAM1"] == 0.12340000
        assert vaspxml.vdw_dft_parameters["PARAM2"] == 1.00000000
        assert vaspxml.vdw_dft_parameters["BPARAM"] == 6.30000000
        assert vaspxml.vdw_dft_parameters["CPARAM"] == 0.00930000

    def test_model_gw_parameters(self) -> None:
        vaspxml = VaspXML.from_str(PARAMETER_ELEMENT)
        assert vaspxml.model_gw_parameters is not None
        assert isinstance(vaspxml.model_gw_parameters, dict)
        assert vaspxml.model_gw_parameters["MODEL_GW"] == 0
        assert vaspxml.model_gw_parameters["MODEL_EPS0"] == 6.82944215
        assert vaspxml.model_gw_parameters["MODEL_ALPHA"] == 1.00000000

    def test_linear_response_parameters(self) -> None:
        vaspxml = VaspXML.from_str(PARAMETER_ELEMENT)
        assert vaspxml.linear_response_parameters is not None
        assert isinstance(vaspxml.linear_response_parameters, dict)
        assert vaspxml.linear_response_parameters["LEPSILON"] is False
        assert vaspxml.linear_response_parameters["LRPA"] is False
        assert vaspxml.linear_response_parameters["LNABLA"] is False
        assert vaspxml.linear_response_parameters["LVEL"] is False
        assert vaspxml.linear_response_parameters["CSHIFT"] == 0.10000000
        assert vaspxml.linear_response_parameters["OMEGAMAX"] == -1.00000000
        assert vaspxml.linear_response_parameters["DEG_THRESHOLD"] == 0.00200000
        assert vaspxml.linear_response_parameters["RTIME"] == -0.10000000
        assert vaspxml.linear_response_parameters["WPLASMAI"] == 0.00000000
        assert np.allclose(vaspxml.linear_response_parameters["DFIELD"], np.array([0.0, 0.0, 0.0]))
        assert np.allclose(
            vaspxml.linear_response_parameters["WPLASMA"],
            np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        )

    def test_parse_orbital_magnetization_parameters(self) -> None:
        vaspxml = VaspXML.from_str(PARAMETER_ELEMENT)
        assert vaspxml.orbital_magnetization_parameters is not None
        assert isinstance(vaspxml.orbital_magnetization_parameters, dict)
        assert vaspxml.orbital_magnetization_parameters["NUCIND"] is False
        assert vaspxml.orbital_magnetization_parameters["LNICSALL"] is True
        assert vaspxml.orbital_magnetization_parameters["ORBITALMAG"] is False
        assert vaspxml.orbital_magnetization_parameters["LMAGBLOCH"] is False
        assert vaspxml.orbital_magnetization_parameters["LCHIMAG"] is False
        assert vaspxml.orbital_magnetization_parameters["LGAUGE"] is True
        assert vaspxml.orbital_magnetization_parameters["MAGATOM"] == 0
        assert np.allclose(
            vaspxml.orbital_magnetization_parameters["MAGDIPOL"], np.array([0.0, 0.0, 0.0])
        )
        assert np.allclose(
            vaspxml.orbital_magnetization_parameters["AVECCONST"], np.array([0.0, 0.0, 0.0])
        )

    def test_response_functions_parameters(self) -> None:
        vaspxml = VaspXML.from_str(PARAMETER_ELEMENT)
        assert vaspxml.response_functions_parameters is not None
        assert isinstance(vaspxml.response_functions_parameters, dict)
        assert vaspxml.response_functions_parameters["LALL_IN_ONE"] is False
        assert vaspxml.response_functions_parameters["IALL_IN_ONE"] == -1
        assert vaspxml.response_functions_parameters["NBANDS_WAVE"] == -1
        assert vaspxml.response_functions_parameters["LFINITE_TEMPERATURE"] is False
        assert vaspxml.response_functions_parameters["LADDER"] is False
        assert vaspxml.response_functions_parameters["LRPAFORCE"] is False
        assert vaspxml.response_functions_parameters["LFXC"] is False
        assert vaspxml.response_functions_parameters["LHARTREE"] is True
        assert vaspxml.response_functions_parameters["IBSE"] == 0
        assert np.array_equal(
            vaspxml.response_functions_parameters["KPOINT"], np.array([-1, 0, 0, 0])
        )
        assert vaspxml.response_functions_parameters["LTCTC"] is False
        assert vaspxml.response_functions_parameters["LTCTE"] is False
        assert vaspxml.response_functions_parameters["LTETE"] is False
        assert vaspxml.response_functions_parameters["LTRIPLET"] is False
        assert vaspxml.response_functions_parameters["LFXCEPS"] is False
        assert vaspxml.response_functions_parameters["LFXHEG"] is False
        assert vaspxml.response_functions_parameters["NATURALO"] == 2
        assert vaspxml.response_functions_parameters["LHOLEGF"] is False
        assert vaspxml.response_functions_parameters["L2ORDER"] is False
        assert vaspxml.response_functions_parameters["LDMP1"] is False
        assert vaspxml.response_functions_parameters["LMP2LT"] is False
        assert vaspxml.response_functions_parameters["LSMP2LT"] is False
        assert vaspxml.response_functions_parameters["LGWLF"] is False
        assert vaspxml.response_functions_parameters["ENCUTGW"] == -1.60000002
        assert vaspxml.response_functions_parameters["ENCUTGWSOFT"] == -1.60000002
        assert vaspxml.response_functions_parameters["ENCUTLF"] == -1.00000000
        assert vaspxml.response_functions_parameters["ESF_SPLINES"] is False
        assert vaspxml.response_functions_parameters["ESF_CONV"] == 0.01000000
        assert vaspxml.response_functions_parameters["ESF_NINTER"] == 15
        assert vaspxml.response_functions_parameters["LMAXMP2"] is -1
        assert vaspxml.response_functions_parameters["SCISSOR"] == 0.00000000
        assert vaspxml.response_functions_parameters["NOMEGA"] == 0
        assert vaspxml.response_functions_parameters["NOMEGAR"] == 0
        assert vaspxml.response_functions_parameters["NBANDSGW"] == -1
        assert vaspxml.response_functions_parameters["NBANDSO"] == -1
        assert vaspxml.response_functions_parameters["NBANDSV"] == -1
        assert vaspxml.response_functions_parameters["NELMGW"] == 1
        assert vaspxml.response_functions_parameters["NELMHF"] == 1
        assert vaspxml.response_functions_parameters["DIM"] == 3
        assert vaspxml.response_functions_parameters["IESPILON"] == 4
        assert vaspxml.response_functions_parameters["ANTIRES"] == 0
        assert vaspxml.response_functions_parameters["OMEGAMAX"] == -30.00000000
        assert vaspxml.response_functions_parameters["OMEGAMIN"] == -30.00000000
        assert vaspxml.response_functions_parameters["OMEGATL"] == -200.00000000
        assert vaspxml.response_functions_parameters["OMEGAGRID"] == 0
        assert vaspxml.response_functions_parameters["CSHIFT"] == -0.10000000
        assert vaspxml.response_functions_parameters["LSELFENERGY"] is False
        assert vaspxml.response_functions_parameters["LSPECTRAL"] is False
        assert vaspxml.response_functions_parameters["LSPECTRALGW"] is False
        assert vaspxml.response_functions_parameters["LSINGLES"] is False
        assert vaspxml.response_functions_parameters["LFERMIGW"] is False
        assert vaspxml.response_functions_parameters["ODDONLYGW"] is False
        assert vaspxml.response_functions_parameters["EVENONLYGW"] is False
        assert vaspxml.response_functions_parameters["NKREDLFX"] == 1
        assert vaspxml.response_functions_parameters["NKREDLFY"] == 1
        assert vaspxml.response_functions_parameters["NKREDLFZ"] == 1
        assert vaspxml.response_functions_parameters["MAXMEM"] == 2800
        assert vaspxml.response_functions_parameters["TELESCOPE"] == 0
        assert vaspxml.response_functions_parameters["NTAUPAR"] == -1
        assert vaspxml.response_functions_parameters["NOMEGAPAR"] == -1
        assert vaspxml.response_functions_parameters["DAMP_NEWTON"] == 0.80000001
        assert vaspxml.response_functions_parameters["LAMBDA"] == 1.00000000

    def test_external_order_field_parameters(self) -> None:
        vaspxml = VaspXML.from_str(PARAMETER_ELEMENT)
        assert vaspxml.external_order_field_parameters is not None
        assert isinstance(vaspxml.external_order_field_parameters, dict)
        assert vaspxml.external_order_field_parameters["OFIELD_KAPPA"] == 0.00000000
        assert np.allclose(
            vaspxml.external_order_field_parameters["OFIELD_K"], np.array([0.0, 0.0, 0.0])
        )
        assert vaspxml.external_order_field_parameters["OFIELD_Q6_NEAR"] == 0.00000000
        assert vaspxml.external_order_field_parameters["OFIELD_Q6_FAR"] == 0.00000000
        assert vaspxml.external_order_field_parameters["OFIELD_A"] == 0.00000000

    def test_optional_k_points_parameters(self) -> None:
        vaspxml = VaspXML.from_str(PARAMETER_ELEMENT)
        assert vaspxml.optional_k_points_parameters is not None
        assert isinstance(vaspxml.optional_k_points_parameters, dict)
        assert vaspxml.optional_k_points_parameters["KPOINTS_OPT_MODE"] == 1
        assert vaspxml.optional_k_points_parameters["LKPOINTS_OPT"] is False


ATOM_INFO = """<atominfo>
  <atoms>       5 </atoms>
  <types>       3 </types>
  <array name="atoms" >
   <dimension dim="1">ion</dimension>
   <field type="string">element</field>
   <field type="int">atomtype</field>
   <set>
    <rc><c>Sr</c><c>   1</c></rc>
    <rc><c>V </c><c>   2</c></rc>
    <rc><c>O </c><c>   3</c></rc>
    <rc><c>O </c><c>   3</c></rc>
    <rc><c>O </c><c>   3</c></rc>
   </set>
  </array>
  <array name="atomtypes" >
   <dimension dim="1">type</dimension>
   <field type="int">atomspertype</field>
   <field type="string">element</field>
   <field>mass</field>
   <field>valence</field>
   <field type="string">pseudopotential</field>
   <set>
    <rc><c>   1</c><c>Sr</c><c>     87.62000000</c><c>     10.00000000</c><c>  PAW_PBE Sr_sv 07Sep2000               </c></rc>
    <rc><c>   1</c><c>V </c><c>     50.94100000</c><c>      5.00000000</c><c>  PAW_PBE V 08Apr2002                   </c></rc>
    <rc><c>   3</c><c>O </c><c>     16.00000000</c><c>      6.00000000</c><c>   PAW_PBE O 08Apr2002                  </c></rc>
   </set>
  </array>
 </atominfo> 
"""


def parse_atom_info_element() -> None:
    _parser = VaspXML.from_str(ATOM_INFO)
    assert False


PRIMITIVE_CELL_ELEMENT = """<primitive_cell>
  <structure name="primitive_cell" >
   <crystal>
    <varray name="basis" >
     <v>       3.84652000       0.00000000       0.00000000 </v>
     <v>       0.00000000       3.84652000       0.00000000 </v>
     <v>       0.00000000       0.00000000       3.84652000 </v>
    </varray>
    <i name="volume">     56.91201793 </i>
    <varray name="rec_basis" >
     <v>       0.25997525       0.00000000       0.00000000 </v>
     <v>       0.00000000       0.25997525       0.00000000 </v>
     <v>       0.00000000       0.00000000       0.25997525 </v>
    </varray>
   </crystal>
   <varray name="positions" >
    <v>       0.00000000       0.00000000       0.00000000 </v>
    <v>       0.50000000       0.50000000       0.50000000 </v>
    <v>       0.50000000       0.50000000       0.00000000 </v>
    <v>       0.50000000       0.00000000       0.50000000 </v>
    <v>       0.00000000       0.50000000       0.50000000 </v>
   </varray>
  </structure>
  <varray name="primitive_index" >
   <v type="int" >        1 </v>
   <v type="int" >        2 </v>
   <v type="int" >        3 </v>
   <v type="int" >        4 </v>
   <v type="int" >        5 </v>
  </varray>
 </primitive_cell>
"""

KPOINTS_ELEMENT = """ <kpoints>
  <generation param="listgenerated">
   <i name="divisions" type="int">      2 </i>
   <v>       0.00000000       0.00000000       0.00000000 </v>
   <v>       0.00000000       0.50000000       0.00000000 </v>
  </generation>
  <varray name="kpointlist" >
   <v>       0.00000000       0.00000000       0.00000000 </v>
   <v>       0.00000000       0.01282051       0.00000000 </v>
  </varray>
  <varray name="weights" >
   <v>       0.00500000 </v>
   <v>       0.00500000 </v>
  </varray>
 </kpoints>
"""


def parse_kpoints_element() -> None:
    parser = VaspXML.from_str(KPOINTS_ELEMENT)

    kpoints = parser.kpoints
    assert kpoints is not None
    assert np.allclose(kpoints.weights, np.array([0.00500000, 0.00500000]))
    assert np.allclose(
        kpoints.kpointlist,
        np.array([[0.00000000, 0.00000000, 0.00000000], [0.00000000, 0.01282051, 0.00000000]]),
    )

    assert kpoints.comment == "listgenerated"
    assert kpoints.mode == "listgenerated"
    assert kpoints.ngrids == 2
    assert kpoints.automatic == False
    assert kpoints.kgrid == [2, 2, 2]
    assert kpoints.kshift == [0, 0, 0]


INCAR_ELEMENT = """<incar>
  <i type="string" name="SYSTEM">Default</i>
  <i type="int" name="ISTART">     0</i>
  <i type="string" name="PREC">Accurate</i>
  <i type="string" name="ALGO">Normal</i>
  <i type="logical" name="ADDGRID"> T  </i>
  <i type="int" name="ISPIN">     1</i>
  <i type="int" name="ICHARG">    11</i>
  <i type="int" name="NELM">   100</i>
  <i type="int" name="NELMIN">     8</i>
  <i name="EDIFF">      0.00000001</i>
  <i name="ENCUT">    600.00000000</i>
  <i type="string" name="LREAL">Auto</i>
  <i type="int" name="ISMEAR">     2</i>
  <i name="SIGMA">      0.20000000</i>
  <i type="int" name="LMAXMIX">     4</i>
  <i type="logical" name="LWAVE"> F  </i>
  <i type="logical" name="LCHARG"> T  </i>
  <i type="int" name="LORBIT">    11</i>
  <i type="logical" name="LASPH"> T  </i>
  <i type="logical" name="LDAU"> T  </i>
  <i type="int" name="LDAUTYPE">     1</i>
  <v type="int" name="LDAUL">    -1     2    -1</v>
  <v name="LDAUU">      0.00000000      5.00000000      0.00000000</v>
  <v name="LDAUJ">      0.00000000      0.00000000      0.00000000</v>
  <i type="int" name="LDAUPRINT">     2</i>
 </incar>
"""


class TestVaspXMLIncar:
    def test_incar_params(self):
        vaspxml = VaspXML.from_str(INCAR_ELEMENT)
        assert vaspxml.incar_parameters is not None
        assert isinstance(vaspxml.incar_parameters, dict)
        assert vaspxml.incar_parameters["SYSTEM"] == "Default"
        assert vaspxml.incar_parameters["ISTART"] == 0
        assert vaspxml.incar_parameters["PREC"] == "Accurate"
        assert vaspxml.incar_parameters["ALGO"] == "Normal"
        assert vaspxml.incar_parameters["ADDGRID"] is True
        assert vaspxml.incar_parameters["ISPIN"] == 1
        assert vaspxml.incar_parameters["ICHARG"] == 11
        assert vaspxml.incar_parameters["NELM"] == 100
        assert vaspxml.incar_parameters["NELMIN"] == 8
        assert vaspxml.incar_parameters["EDIFF"] == 0.00000001
        assert vaspxml.incar_parameters["ENCUT"] == 600.00000000
        assert vaspxml.incar_parameters["LREAL"] == "Auto"
        assert vaspxml.incar_parameters["ISMEAR"] == 2
        assert vaspxml.incar_parameters["SIGMA"] == 0.20000000
        assert vaspxml.incar_parameters["LMAXMIX"] == 4
        assert vaspxml.incar_parameters["LWAVE"] is False
        assert vaspxml.incar_parameters["LCHARG"] is True
        assert vaspxml.incar_parameters["LORBIT"] == 11
        assert vaspxml.incar_parameters["LASPH"] is True
        assert vaspxml.incar_parameters["LDAU"] is True
        assert vaspxml.incar_parameters["LDAUTYPE"] == 1
        assert vaspxml.incar_parameters["LDAUL"] == [-1, 2, -1]
        assert vaspxml.incar_parameters["LDAUU"] == [0.00000000, 5.00000000, 0.00000000]
        assert vaspxml.incar_parameters["LDAUJ"] == [0.00000000, 0.00000000, 0.00000000]
        assert vaspxml.incar_parameters["LDAUPRINT"] == 2


GENERATOR_ELEMENT = """<generator>
  <i name="program" type="string">vasp </i>
  <i name="version" type="string">6.4.3  </i>
  <i name="subversion" type="string">19Mar24 (build Nov 30 2024 18:10:10) complex                          parallel </i>
  <i name="platform" type="string">LinuxIFC </i>
  <i name="date" type="string">2025 03 10 </i>
  <i name="time" type="string">13:01:41 </i>
 </generator>
"""


def test_parse_generator_element():
    vaspxml = VaspXML.from_str(GENERATOR_ELEMENT)
    assert vaspxml.generator_parameters is not None

    assert vaspxml.generator_parameters.program == "vasp"
    assert vaspxml.generator_parameters.version == "6.4.3"
    assert (
        vaspxml.generator_parameters.subversion
        == "19Mar24 (build Nov 30 2024 18:10:10) complex                          parallel"
    )
    assert vaspxml.generator_parameters.platform == "LinuxIFC"
    assert vaspxml.generator_parameters.date == "2025 03 10"
    assert vaspxml.generator_parameters.time == "13:01:41"


STRUCTURE_ELEMENT = """ <structure name="initialpos" >
  <crystal>
   <varray name="basis" >
    <v>       3.84652000       0.00000000       0.00000000 </v>
    <v>       0.00000000       3.84652000       0.00000000 </v>
    <v>       0.00000000       0.00000000       3.84652000 </v>
   </varray>
   <i name="volume">     56.91201793 </i>
   <varray name="rec_basis" >
    <v>       0.25997525       0.00000000       0.00000000 </v>
    <v>       0.00000000       0.25997525       0.00000000 </v>
    <v>       0.00000000       0.00000000       0.25997525 </v>
   </varray>
  </crystal>
  <varray name="positions" >
   <v>       0.00000000       0.00000000       0.00000000 </v>
   <v>       0.50000000       0.50000000       0.50000000 </v>
   <v>       0.50000000       0.50000000       0.00000000 </v>
   <v>       0.50000000       0.00000000       0.50000000 </v>
   <v>       0.00000000       0.50000000       0.50000000 </v>
  </varray>
 </structure>
"""


def test_parse_structure_element():
    vaspxml = VaspXML.from_str(STRUCTURE_ELEMENT)
    assert vaspxml.structure_element is not None
    assert isinstance(vaspxml.structure_element, dict)

    assert vaspxml.structure_element["name"] == "initialpos"

    assert vaspxml.structure_element["crystal"]["basis"] == [
        [3.84652000, 0.00000000, 0.00000000],
        [0.00000000, 3.84652000, 0.00000000],
        [0.00000000, 0.00000000, 3.84652000],
    ]
    assert vaspxml.structure_element["crystal"]["volume"] == 56.91201793
    assert vaspxml.structure_element["crystal"]["rec_basis"] == [
        [0.25997525, 0.00000000, 0.00000000],
        [0.00000000, 0.25997525, 0.00000000],
        [0.00000000, 0.00000000, 0.25997525],
    ]
    assert vaspxml.structure_element["positions"] == [
        [0.00000000, 0.00000000, 0.00000000],
        [0.50000000, 0.50000000, 0.50000000],
        [0.50000000, 0.50000000, 0.00000000],
        [0.50000000, 0.00000000, 0.50000000],
        [0.00000000, 0.50000000, 0.50000000],
    ]


CALCULATION_ELEMENT = """ <calculation>
  <scstep>
   <time name="dav">    1.53    1.62</time>
   <time name="total">    1.78    1.90</time>
   <energy>
    <i name="alphaZ">    184.93671151 </i>
    <i name="ewald">  -2047.90029510 </i>
    <i name="hartreedc">   -429.14745404 </i>
    <i name="XCdc">     76.60242166 </i>
    <i name="pawpsdc">   3744.26834168 </i>
    <i name="pawaedc">  -3690.18265525 </i>
    <i name="eentropy">     -0.00024940 </i>
    <i name="bandstr">    134.90076094 </i>
    <i name="atom">   2310.98487007 </i>
    <i name="e_fr_energy">    284.46245206 </i>
    <i name="e_wo_entrp">    284.46270145 </i>
    <i name="e_0_energy">    284.46251440 </i>
   </energy>
  </scstep>
  <scstep>
   <time name="dav">    1.66    1.70</time>
   <time name="total">    1.66    1.70</time>
   <energy>
    <i name="e_fr_energy">      1.16635921 </i>
    <i name="e_wo_entrp">      1.17056558 </i>
    <i name="e_0_energy">      1.16741080 </i>
   </energy>
  </scstep>
  <scstep>
   <time name="dav">    1.83    1.86</time>
   <time name="total">    1.83    1.86</time>
   <energy>
    <i name="e_fr_energy">    -29.93116027 </i>
    <i name="e_wo_entrp">    -29.93584450 </i>
    <i name="e_0_energy">    -29.93233133 </i>
   </energy>
  </scstep>
  <scstep>
   <time name="dav">    1.94    1.96</time>
   <time name="total">    1.94    1.96</time>
   <energy>
    <i name="e_fr_energy">    -31.26955379 </i>
    <i name="e_wo_entrp">    -31.27198186 </i>
    <i name="e_0_energy">    -31.27016080 </i>
   </energy>
  </scstep>
  <scstep>
   <time name="dav">    1.91    1.93</time>
   <time name="total">    1.91    1.93</time>
   <energy>
    <i name="e_fr_energy">    -31.30890966 </i>
    <i name="e_wo_entrp">    -31.31132993 </i>
    <i name="e_0_energy">    -31.30951473 </i>
   </energy>
  </scstep>
  <scstep>
   <time name="dav">    1.82    1.85</time>
   <time name="total">    1.83    1.85</time>
   <energy>
    <i name="e_fr_energy">    -31.30989551 </i>
    <i name="e_wo_entrp">    -31.31231421 </i>
    <i name="e_0_energy">    -31.31050018 </i>
   </energy>
  </scstep>
  <scstep>
   <time name="dav">    1.91    1.96</time>
   <time name="total">    1.91    1.96</time>
   <energy>
    <i name="e_fr_energy">    -31.30993094 </i>
    <i name="e_wo_entrp">    -31.31234955 </i>
    <i name="e_0_energy">    -31.31053559 </i>
   </energy>
  </scstep>
  <scstep>
   <time name="dav">    1.90    1.93</time>
   <time name="total">    1.90    1.93</time>
   <energy>
    <i name="e_fr_energy">    -31.30993215 </i>
    <i name="e_wo_entrp">    -31.31235076 </i>
    <i name="e_0_energy">    -31.31053680 </i>
   </energy>
  </scstep>
  <scstep>
   <time name="dav">    1.59    1.60</time>
   <time name="total">    1.60    1.60</time>
   <energy>
    <i name="e_fr_energy">    -31.30993216 </i>
    <i name="e_wo_entrp">    -31.31235077 </i>
    <i name="e_0_energy">    -31.31053681 </i>
   </energy>
  </scstep>
  <scstep>
   <time name="dav">    1.14    1.14</time>
   <time name="total">    1.14    1.14</time>
   <energy>
    <i name="alphaZ">    184.93671151 </i>
    <i name="ewald">  -2047.90029510 </i>
    <i name="hartreedc">   -429.14745404 </i>
    <i name="XCdc">     76.60242166 </i>
    <i name="pawpsdc">   3744.26834168 </i>
    <i name="pawaedc">  -3690.18265525 </i>
    <i name="eentropy">      0.00241860 </i>
    <i name="bandstr">   -180.87429128 </i>
    <i name="atom">   2310.98487007 </i>
    <i name="e_fr_energy">    -31.30993216 </i>
    <i name="e_wo_entrp">    -31.31235077 </i>
    <i name="e_0_energy">    -31.31053681 </i>
   </energy>
  </scstep>
 </calculation>
"""


def parse_calculation_self_consistent_steps():
    parser = VaspXML.from_str(CALCULATION_ELEMENT)
    assert len(parser.self_consistent_steps) == 10

    for step in parser.self_consistent_steps:
        assert isinstance(step.time.dav, float)
        assert isinstance(step.time.total, float)
        assert isinstance(step.energy.e_fr_energy, float)
        assert isinstance(step.energy.e_wo_entrp, float)
        assert isinstance(step.energy.e_0_energy, float)

    initial_step = parser.self_consistent_steps[0]
    assert initial_step.time.dav == 1.53
    assert initial_step.time.total == 1.78
    assert initial_step.energy.alphaZ == 184.93671151
    assert initial_step.energy.ewald == -2047.90029510
    assert initial_step.energy.hartreedc == -429.14745404
    assert initial_step.energy.XCdc == 76.60242166
    assert initial_step.energy.pawpsdc == 3744.26834168
    assert initial_step.energy.pawaedc == -3690.18265525
    assert initial_step.energy.eentropy == -0.00024940
    assert initial_step.energy.bandstr == 134.90076094
    assert initial_step.energy.atom == 2310.98487007
    assert initial_step.energy.e_fr_energy == 184.93671151
    assert initial_step.energy.e_wo_entrp == 184.93671151
    assert initial_step.energy.e_0_energy == 184.93671151

    final_step = parser.self_consistent_steps[-1]
    assert final_step.time.dav == 1.14
    assert final_step.time.total == 1.14
    assert final_step.energy.alphaZ == 184.93671151
    assert final_step.energy.ewald == -2047.90029510
    assert final_step.energy.hartreedc == -429.14745404
    assert final_step.energy.XCdc == 76.60242166
    assert final_step.energy.pawpsdc == 3744.26834168
    assert final_step.energy.pawaedc == -3690.18265525
    assert final_step.energy.eentropy == 0.00241860
    assert final_step.energy.bandstr == -180.87429128
    assert final_step.energy.atom == 2310.98487007
    assert final_step.energy.e_fr_energy == -31.30993216
    assert final_step.energy.e_wo_entrp == -31.31235077
    assert final_step.energy.e_0_energy == -31.31053681


INITIAL_STRUCTURE_ELEMENT = """ <calculation> <structure>
   <crystal>
    <varray name="basis" >
     <v>       3.84652000       0.00000000       0.00000000 </v>
     <v>       0.00000000       3.84652000       0.00000000 </v>
     <v>       0.00000000       0.00000000       3.84652000 </v>
    </varray>
    <i name="volume">     56.91201793 </i>
    <varray name="rec_basis" >
     <v>       0.25997525       0.00000000       0.00000000 </v>
     <v>       0.00000000       0.25997525       0.00000000 </v>
     <v>       0.00000000       0.00000000       0.25997525 </v>
    </varray>
   </crystal>
   <varray name="positions" >
    <v>       0.00000000       0.00000000       0.00000000 </v>
    <v>       0.50000000       0.50000000       0.50000000 </v>
   </varray>
  </structure>
  </calculation>
"""


def parse_initial_structure_element() -> None:
    parser = VaspXML.from_str(INITIAL_STRUCTURE_ELEMENT)
    initial_structure = parser.initial_structure
    assert initial_structure is not None
    assert initial_structure.crystal.basis.shape == (3, 3)
    assert initial_structure.crystal.volume == 56.91201793
    assert initial_structure.crystal.rec_basis.shape == (3, 3)
    assert initial_structure.positions.shape == (2, 3)
    assert np.allclose(
        initial_structure.positions,
        np.array([[0.00000000, 0.00000000, 0.00000000], [0.50000000, 0.50000000, 0.50000000]]),
    )


FORCES_ELEMENT = """ <calculation> <forces>
   <varray name="forces" >
    <v>      -0.00000000      -5.00000000      -0.00000000 </v>
    <v>      -0.00000000      -5.00000000      -5.00000000 </v>
   </varray>
  </forces>
 </calculation>
"""


def parse_forces_element() -> None:
    parser = VaspXML.from_str(FORCES_ELEMENT)
    forces = parser.forces
    assert forces is not None
    assert forces.shape == (2, 3)
    assert np.allclose(
        forces,
        np.array(
            [[-0.00000000, -5.00000000, -0.00000000], [-0.00000000, -5.00000000, -5.00000000]]
        ),
    )


STRESS_ELEMENT = """ <calculation> <stress>
   <varray name="stress" >
    <v>     229.62291361       0.00000000      -0.00000000 </v>
    <v>       0.00000000     229.62291361       0.00000000 </v>
    <v>       0.00000000       0.00000000     229.62291361 </v>
   </varray>
  </stress>
 </calculation>
"""


def parse_stress_element() -> None:
    parser = VaspXML.from_str(STRESS_ELEMENT)
    stress = parser.stress
    assert stress is not None
    assert np.allclose(
        stress,
        np.array(
            [
                [229.62291361, 0.00000000, -0.00000000],
                [0.00000000, 229.62291361, 0.00000000],
                [0.00000000, 0.00000000, 229.62291361],
            ]
        ),
    )


ENERGY_ELEMENT = """ <calculation> <energy>
   <i name="e_fr_energy">    -31.30993216 </i>
   <i name="e_wo_entrp">    -31.31235077 </i>
   <i name="e_0_energy">    -31.31053681 </i>
  </energy>
 </calculation>
"""


def parse_energy_element():
    parser = VaspXML.from_str(ENERGY_ELEMENT)
    assert parser.e_fr_energy == -31.30993216
    assert parser.e_wo_entrp == -31.31235077
    assert parser.e_0_energy == -31.31053681


NON_SPIN_POLARIZED_EIGENVALUES_ELEMENT = """<eigenvalues>
   <array>
    <dimension dim="1">band</dimension>
    <dimension dim="2">kpoint</dimension>
    <dimension dim="3">spin</dimension>
    <field>eigene</field>
    <field>occ</field>
    <set>
     <set comment="spin 1">
      <set comment="kpoint 1">
       <r>  -29.0611    1.0000 </r>
       <r>  -14.3001    1.0000 </r>
      </set>
      <set comment="kpoint 2">
       <r>  -29.0610    1.0000 </r>
       <r>  -14.2999    1.0000 </r>
      </set>
     </set>
    </set>
   </array>
</eigenvalues>
"""


SPIN_POLARIZED_CONVERGED_EIGENVALUES_ELEMENT = """ <eigenvalues>
   <array>
    <dimension dim="1">band</dimension>
    <dimension dim="2">kpoint</dimension>
    <dimension dim="3">spin</dimension>
    <field>eigene</field>
    <field>occ</field>
    <set>
     <set comment="spin 1">
      <set comment="kpoint 1">
       <r>  -29.0715    1.0000 </r>
       <r>  -14.2137    1.0000 </r>
       </set>
      <set comment="kpoint 2">
       <r>  -29.0715    1.0000 </r>
       <r>  -14.2136    1.0000 </r>
       </set>
     </set>
    <set comment="spin 2">
      <set comment="kpoint 1">
       <r>  -29.0750    1.0000 </r>
       <r>  -14.3186    1.0000 </r>
       </set>
      <set comment="kpoint 2">
       <r>  -29.0715    1.0000 </r>
       <r>  -14.2136    1.0000 </r>
       </set>
     </set>
    </set>
   </array>
  </eigenvalues>
"""


class VaspXMLEigenvaluesTestCase(NamedTuple):
    data: str
    id: str
    n_spins: int


eigenvalues_test_cases = [
    VaspXMLEigenvaluesTestCase(
        data=NON_SPIN_POLARIZED_EIGENVALUES_ELEMENT, id="non_spin_polarized_eigenvalues", n_spins=1
    ),
    VaspXMLEigenvaluesTestCase(
        data=SPIN_POLARIZED_CONVERGED_EIGENVALUES_ELEMENT,
        id="spin_polarized_eigenvalues",
        n_spins=2,
    ),
]


class TestVasprunEigenvalues:
    @pytest.fixture(params=eigenvalues_test_cases, ids=lambda c: c.id)  # pyright: ignore[reportUnknownLambdaType, reportUnknownMemberType]
    def case(self, request: pytest.FixtureRequest) -> VaspXMLEigenvaluesTestCase:
        return request.param

    def test_eigenvalues_shape(self, case: VaspXMLEigenvaluesTestCase) -> None:
        vaspxml = VaspXML.from_str(case.data)
        eigenvalues = vaspxml.eigenvalues
        assert eigenvalues is not None
        assert eigenvalues.shape == (2, 2, case.n_spins)


NON_SPIN_POLARIZED_TOTAL_DOS_ELEMENT = """ <total>
    <array>
     <dimension dim="1">gridpoints</dimension>
     <dimension dim="2">spin</dimension>
     <field>energy</field>
     <field>total</field>
     <field>integrated</field>
     <set>
      <set comment="spin 1">
       <r>   -31.0611     0.0000     0.0000 </r>
       <r>   -30.9153     0.0000     0.0000 </r>
      </set>
     </set>
    </array>
   </total>
"""

SPIN_POLARIZED_TOTAL_DOS_ELEMENT = """ <total>
    <array>
     <dimension dim="1">gridpoints</dimension>
     <dimension dim="2">spin</dimension>
     <field>energy</field>
     <field>total</field>
     <field>integrated</field>
     <set>
      <set comment="spin 1">
       <r>   -29.7471     0.0001     0.0000 </r>
       <r>   -29.5623     0.0076     0.0014 </r>
      </set>
      <set comment="spin 2">
       <r>   -31.5954     0.0000     0.0000 </r>
       <r>   -31.4105     0.0000     0.0000 </r>
     </set>
     </set>
    </array>
   </total>
"""

NON_COLINEAR_TOTAL_DOS_ELEMENT = """
   <total>
    <array>
     <dimension dim="1">gridpoints</dimension>
     <dimension dim="2">spin</dimension>
     <field>energy</field>
     <field>total</field>
     <field>integrated</field>
     <set>
      <set comment="spin 1">
       <r>   -29.7471     0.0001     0.0000 </r>
       <r>   -29.5623     0.0076     0.0014 </r>
      </set>
      <set comment="spin 2">
       <r>   -31.5954     0.0000     0.0000 </r>
       <r>   -31.4105     0.0000     0.0000 </r>
     </set>
     <set comment="spin 3">
       <r>   -31.5954     0.0000     0.0000 </r>
       <r>   -31.4105     0.0000     0.0000 </r>
     </set>
     <set comment="spin 4">
       <r>   -31.5954     0.0000     0.0000 </r>
       <r>   -31.4105     0.0000     0.0000 </r>
     </set>
    </set>
  </array>
</total>
"""


class VaspXMLTotalTestCase(NamedTuple):
    data: str
    id: str
    n_spins: int


total_test_cases = [
    VaspXMLTotalTestCase(
        data=NON_SPIN_POLARIZED_TOTAL_DOS_ELEMENT, id="non_spin_polarized_total", n_spins=1
    ),
    VaspXMLTotalTestCase(
        data=SPIN_POLARIZED_TOTAL_DOS_ELEMENT, id="spin_polarized_total", n_spins=2
    ),
    VaspXMLTotalTestCase(data=NON_COLINEAR_TOTAL_DOS_ELEMENT, id="non_colinear_total", n_spins=4),
]


class TestVasprunTotal:
    @pytest.fixture(params=total_test_cases, ids=lambda c: c.id)  # pyright: ignore[reportUnknownLambdaType, reportUnknownMemberType]
    def case(self, request: pytest.FixtureRequest) -> VaspXMLTotalTestCase:
        return request.param

    def test_total_shape(self, case: VaspXMLTotalTestCase) -> None:
        vaspxml = VaspXML.from_str(case.data)
        total = vaspxml.total
        assert total is not None
        assert total.shape == (2, case.n_spins)


NON_SPIN_POLARIZED_PARTIAL_DOS_ELEMENT = """  <partial>
    <array>
     <dimension dim="1">gridpoints</dimension>
     <dimension dim="2">spin</dimension>
     <dimension dim="3">ion</dimension>
     <field>energy</field>
     <field>    s</field>
     <field>   py</field>
     <field>   pz</field>
     <field>   px</field>
     <field>  dxy</field>
     <field>  dyz</field>
     <field>  dz2</field>
     <field>  dxz</field>
     <field>x2-y2</field>
     <set>
      <set comment="ion 1">
       <set comment="spin 1">
        <r>   -31.0611     0.0000     0.0000     0.0000     0.0000     0.0000     0.0000     0.0000     0.0000     0.0000 </r>
        <r>   -30.9153     0.0000     0.0000     0.0000     0.0000     0.0000     0.0000     0.0000     0.0000     0.0000 </r>
    </set>
      </set>
      <set comment="ion 2">
       <set comment="spin 1">
        <r>   -31.0611     0.0000     0.0000     0.0000     0.0000     0.0000     0.0000     0.0000     0.0000     0.0000 </r>
        <r>   -30.9153     0.0000     0.0000     0.0000     0.0000     0.0000     0.0000     0.0000     0.0000     0.0000 </r>
       </set>
      </set>
     </set>
    </array>
   </partial>
"""

SPIN_POLARIZED_PARTIAL_DOS_ELEMENT = """<partial>
    <array>
     <dimension dim="1">gridpoints</dimension>
     <dimension dim="2">spin</dimension>
     <dimension dim="3">ion</dimension>
     <field>energy</field>
     <field>    s</field>
     <field>   py</field>
     <field>   pz</field>
     <field>   px</field>
     <field>  dxy</field>
     <field>  dyz</field>
     <field>  dz2</field>
     <field>  dxz</field>
     <field>x2-y2</field>
     <set>
      <set comment="ion 1">
       <set comment="spin 1">
        <r>   -31.5954     0.0000     0.0000     0.0000     0.0000     0.0000     0.0000     0.0000     0.0000     0.0000 </r>
        <r>   -31.4105     0.0000     0.0000     0.0000     0.0000     0.0000     0.0000     0.0000     0.0000     0.0000 </r>
      </set>
       <set comment="spin 2">
        <r>   -31.5954     0.0000     0.0000     0.0000     0.0000     0.0000     0.0000     0.0000     0.0000     0.0000 </r>
        <r>   -31.4105     0.0000     0.0000     0.0000     0.0000     0.0000     0.0000     0.0000     0.0000     0.0000 </r>
    </set>
      </set>
      <set comment="ion 2">
       <set comment="spin 1">
        <r>   -31.5954     0.0000     0.0000     0.0000     0.0000     0.0000     0.0000     0.0000     0.0000     0.0000 </r>
        <r>   -31.4105     0.0000     0.0000     0.0000     0.0000     0.0000     0.0000     0.0000     0.0000     0.0000 </r>
        </set>
       <set comment="spin 2">
        <r>   -31.5954     0.0000     0.0000     0.0000     0.0000     0.0000     0.0000     0.0000     0.0000     0.0000 </r>
        <r>   -31.4105     0.0000     0.0000     0.0000     0.0000     0.0000     0.0000     0.0000     0.0000     0.0000 </r>
      </set>
      </set>
     </set>
    </array>
   </partial>
"""


NON_COLINEAR_PARTIAL_DOS_ELEMENT = """<partial>
    <array>
     <dimension dim="1">gridpoints</dimension>
     <dimension dim="2">spin</dimension>
     <dimension dim="3">ion</dimension>
     <field>energy</field>
     <field>    s</field>
     <field>   py</field>
     <field>   pz</field>
     <field>   px</field>
     <field>  dxy</field>
     <field>  dyz</field>
     <field>  dz2</field>
     <field>  dxz</field>
     <field>x2-y2</field>
     <set>
      <set comment="ion 1">
       <set comment="spin 1">
        <r>   -31.5954     0.0000     0.0000     0.0000     0.0000     0.0000     0.0000     0.0000     0.0000     0.0000 </r>
        <r>   -31.4105     0.0000     0.0000     0.0000     0.0000     0.0000     0.0000     0.0000     0.0000     0.0000 </r>
      </set>
       <set comment="spin 2">
        <r>   -31.5954     0.0000     0.0000     0.0000     0.0000     0.0000     0.0000     0.0000     0.0000     0.0000 </r>
        <r>   -31.4105     0.0000     0.0000     0.0000     0.0000     0.0000     0.0000     0.0000     0.0000     0.0000 </r>
    </set>
    <set comment="spin 3">
        <r>   -31.5954     0.0000     0.0000     0.0000     0.0000     0.0000     0.0000     0.0000     0.0000     0.0000 </r>
        <r>   -31.4105     0.0000     0.0000     0.0000     0.0000     0.0000     0.0000     0.0000     0.0000     0.0000 </r>
      </set>
      <set comment="spin 4">
        <r>   -31.5954     0.0000     0.0000     0.0000     0.0000     0.0000     0.0000     0.0000     0.0000     0.0000 </r>
        <r>   -31.4105     0.0000     0.0000     0.0000     0.0000     0.0000     0.0000     0.0000     0.0000     0.0000 </r>
      </set>
      </set>
      <set comment="ion 2">
       <set comment="spin 1">
        <r>   -31.5954     0.0000     0.0000     0.0000     0.0000     0.0000     0.0000     0.0000     0.0000     0.0000 </r>
        <r>   -31.4105     0.0000     0.0000     0.0000     0.0000     0.0000     0.0000     0.0000     0.0000     0.0000 </r>
        </set>
       <set comment="spin 2">
        <r>   -31.5954     0.0000     0.0000     0.0000     0.0000     0.0000     0.0000     0.0000     0.0000     0.0000 </r>
        <r>   -31.4105     0.0000     0.0000     0.0000     0.0000     0.0000     0.0000     0.0000     0.0000     0.0000 </r>
      </set>
      <set comment="spin 3">
        <r>   -31.5954     0.0000     0.0000     0.0000     0.0000     0.0000     0.0000     0.0000     0.0000     0.0000 </r>
        <r>   -31.4105     0.0000     0.0000     0.0000     0.0000     0.0000     0.0000     0.0000     0.0000     0.0000 </r>
      </set>
      <set comment="spin 4">
        <r>   -31.5954     0.0000     0.0000     0.0000     0.0000     0.0000     0.0000     0.0000     0.0000     0.0000 </r>
        <r>   -31.4105     0.0000     0.0000     0.0000     0.0000     0.0000     0.0000     0.0000     0.0000     0.0000 </r>
      </set>
      </set>
     </set>
    </array>
   </partial>
"""


class VaspXMLPartialTestCase(NamedTuple):
    data: str
    id: str
    n_spins: int


partial_test_cases = [
    VaspXMLPartialTestCase(
        data=NON_SPIN_POLARIZED_PARTIAL_DOS_ELEMENT, id="non_spin_polarized_partial", n_spins=1
    ),
    VaspXMLPartialTestCase(
        data=SPIN_POLARIZED_PARTIAL_DOS_ELEMENT, id="spin_polarized_partial", n_spins=2
    ),
    VaspXMLPartialTestCase(
        data=NON_COLINEAR_PARTIAL_DOS_ELEMENT, id="non_colinear_partial", n_spins=4
    ),
]


class TestVasprunPartial:
    @pytest.fixture(params=partial_test_cases, ids=lambda c: c.id)  # pyright: ignore[reportUnknownLambdaType, reportUnknownMemberType]
    def case(self, request: pytest.FixtureRequest) -> VaspXMLPartialTestCase:
        return request.param

    def test_partial_shape(self, case: VaspXMLPartialTestCase) -> None:
        vaspxml = VaspXML.from_str(case.data)
        partial = vaspxml.partial
        assert partial is not None
        assert partial.shape == (2, case.n_spins, 2, 9)


NON_SPIN_POLARIZED_PROJ_ELEMENT = """ <projected>
   <eigenvalues>
    <array>
     <dimension dim="1">band</dimension>
     <dimension dim="2">kpoint</dimension>
     <dimension dim="3">spin</dimension>
     <field>eigene</field>
     <field>occ</field>
     <set>
      <set comment="spin 1">
       <set comment="kpoint 1">
        <r>  -29.0611    1.0000 </r>
        <r>  -14.3001    1.0000 </r>
      </set>
       <set comment="kpoint 2">
        <r>  -29.0610    1.0000 </r>
        <r>  -14.2999    1.0000 </r>
      </set>
      </set>
     </set>
    </array>
   </eigenvalues>
    <array>
    <dimension dim="1">ion</dimension>
    <dimension dim="2">band</dimension>
    <dimension dim="3">kpoint</dimension>
    <dimension dim="4">spin</dimension>
    <field>    s</field>
    <field>   py</field>
    <field>   pz</field>
    <field>   px</field>
    <field>  dxy</field>
    <field>  dyz</field>
    <field>  dz2</field>
    <field>  dxz</field>
    <field>x2-y2</field>
    <set>
     <set comment="spin1">
      <set comment="kpoint 1">
       <set comment="band 1">
        <r>  0.9723  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000 </r>
        <r>  0.0005  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000 </r>
       </set>
       <set comment="band 2">
        <r>  0.0108  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000 </r>
        <r>  0.1372  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000 </r>
       </set>
       </set>
      <set comment="kpoint 2">
       <set comment="band 1">
        <r>  0.9723  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000 </r>
        <r>  0.0005  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000 </r>
       </set>
       <set comment="band 2">
        <r>  0.0108  0.0002  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000 </r>
        <r>  0.1371  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000 </r>
       </set>
    </set>
    </set>
    </set>
   </array>
</projected>
"""


SPIN_POLARIZED_PROJ_ELEMENT = """ <projected>
   <eigenvalues>
    <array>
     <dimension dim="1">band</dimension>
     <dimension dim="2">kpoint</dimension>
     <dimension dim="3">spin</dimension>
     <field>eigene</field>
     <field>occ</field>
     <set>
      <set comment="spin 1">
       <set comment="kpoint 1">
        <r>  -29.0715    1.0000 </r>
        <r>  -14.2137    1.0000 </r>
        </set>
       <set comment="kpoint 2">
        <r>  -29.0715    1.0000 </r>
        <r>  -14.2136    1.0000 </r>
        </set>
      </set>
      <set comment="spin 2">
       <set comment="kpoint 1">
        <r>  -29.0750    1.0000 </r>
        <r>  -14.3186    1.0000 </r>
        </set>
       <set comment="kpoint 2">
        <r>  -29.0715    1.0000 </r>
        <r>  -14.2136    1.0000 </r>
        </set>
      </set>
     </set>
    </array>
   </eigenvalues>
   <array>
    <dimension dim="1">ion</dimension>
    <dimension dim="2">band</dimension>
    <dimension dim="3">kpoint</dimension>
    <dimension dim="4">spin</dimension>
    <field>    s</field>
    <field>   py</field>
    <field>   pz</field>
    <field>   px</field>
    <field>  dxy</field>
    <field>  dyz</field>
    <field>  dz2</field>
    <field>  dxz</field>
    <field>x2-y2</field>
    <set>
     <set comment="spin1">
      <set comment="kpoint 1">
       <set comment="band 1">
        <r>  0.9724  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000 </r>
        <r>  0.0005  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000 </r>
        </set>
       <set comment="band 2">
        <r>  0.0109  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000 </r>
        <r>  0.1397  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000 </r>
       </set>
      </set>
      <set comment="kpoint 2">
       <set comment="band 1">
        <r>  0.9724  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000 </r>
        <r>  0.0005  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000 </r>
       </set>
       <set comment="band 2">
        <r>  0.0108  0.0002  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000 </r>
        <r>  0.1396  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000 </r>
        </set>
      </set>
     </set>
     <set comment="spin2">
      <set comment="kpoint 1">
       <set comment="band 1">
        <r>  0.9724  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000 </r>
        <r>  0.0005  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000 </r>
        </set>
       <set comment="band 2">
        <r>  0.0109  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000 </r>
        <r>  0.1397  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000 </r>
       </set>
      </set>
      <set comment="kpoint 2">
       <set comment="band 1">
        <r>  0.9724  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000 </r>
        <r>  0.0005  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000 </r>
       </set>
       <set comment="band 2">
        <r>  0.0108  0.0002  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000 </r>
        <r>  0.1396  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000 </r>
        </set>
      </set>
     </set>
    </set>
   </array>
  </projected>
"""

NON_COLINEAR_PROJ_ELEMENT = """ <projected>
   <eigenvalues>
    <array>
     <dimension dim="1">band</dimension>
     <dimension dim="2">kpoint</dimension>
     <dimension dim="3">spin</dimension>
     <field>eigene</field>
     <field>occ</field>
     <set>
      <set comment="spin 1">
       <set comment="kpoint 1">
        <r>  -29.0715    1.0000 </r>
        <r>  -14.2137    1.0000 </r>
        </set>
       <set comment="kpoint 2">
        <r>  -29.0715    1.0000 </r>
        <r>  -14.2136    1.0000 </r>
        </set>
      </set>
      <set comment="spin 2">
       <set comment="kpoint 1">
        <r>  -29.0750    1.0000 </r>
        <r>  -14.3186    1.0000 </r>
        </set>
       <set comment="kpoint 2">
        <r>  -29.0715    1.0000 </r>
        <r>  -14.2136    1.0000 </r>
        </set>
      </set>
      <set comment="spin 3">
       <set comment="kpoint 1">
        <r>  -29.0750    1.0000 </r>
        <r>  -14.3186    1.0000 </r>
        </set>
       <set comment="kpoint 2">
        <r>  -29.0715    1.0000 </r>
        <r>  -14.2136    1.0000 </r>
        </set>
      </set>
      <set comment="spin 4">
       <set comment="kpoint 1">
        <r>  -29.0750    1.0000 </r>
        <r>  -14.3186    1.0000 </r>
        </set>
       <set comment="kpoint 2">
        <r>  -29.0715    1.0000 </r>
        <r>  -14.2136    1.0000 </r>
        </set>
      </set>
     </set>
    </array>
   </eigenvalues>
   <array>
    <dimension dim="1">ion</dimension>
    <dimension dim="2">band</dimension>
    <dimension dim="3">kpoint</dimension>
    <dimension dim="4">spin</dimension>
    <field>    s</field>
    <field>   py</field>
    <field>   pz</field>
    <field>   px</field>
    <field>  dxy</field>
    <field>  dyz</field>
    <field>  dz2</field>
    <field>  dxz</field>
    <field>x2-y2</field>
    <set>
     <set comment="spin1">
      <set comment="kpoint 1">
       <set comment="band 1">
        <r>  0.9724  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000 </r>
        <r>  0.0005  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000 </r>
        </set>
       <set comment="band 2">
        <r>  0.0109  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000 </r>
        <r>  0.1397  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000 </r>
       </set>
      </set>
      <set comment="kpoint 2">
       <set comment="band 1">
        <r>  0.9724  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000 </r>
        <r>  0.0005  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000 </r>
       </set>
       <set comment="band 2">
        <r>  0.0108  0.0002  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000 </r>
        <r>  0.1396  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000 </r>
        </set>
      </set>
     </set>
     <set comment="spin2">
      <set comment="kpoint 1">
       <set comment="band 1">
        <r>  0.9724  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000 </r>
        <r>  0.0005  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000 </r>
        </set>
       <set comment="band 2">
        <r>  0.0109  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000 </r>
        <r>  0.1397  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000 </r>
       </set>
      </set>
      <set comment="kpoint 2">
       <set comment="band 1">
        <r>  0.9724  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000 </r>
        <r>  0.0005  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000 </r>
       </set>
       <set comment="band 2">
        <r>  0.0108  0.0002  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000 </r>
        <r>  0.1396  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000 </r>
        </set>
      </set>
     </set>
     
    <set comment="spin3">
      <set comment="kpoint 1">
       <set comment="band 1">
        <r>  0.9724  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000 </r>
        <r>  0.0005  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000 </r>
        </set>
       <set comment="band 2">
        <r>  0.0109  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000 </r>
        <r>  0.1397  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000 </r>
       </set>
      </set>
      <set comment="kpoint 2">
       <set comment="band 1">
        <r>  0.9724  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000 </r>
        <r>  0.0005  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000 </r>
       </set>
       <set comment="band 2">
        <r>  0.0108  0.0002  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000 </r>
        <r>  0.1396  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000 </r>
        </set>
      </set>
     </set>
     <set comment="spin4">
      <set comment="kpoint 1">
       <set comment="band 1">
        <r>  0.9724  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000 </r>
        <r>  0.0005  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000 </r>
        </set>
       <set comment="band 2">
        <r>  0.0109  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000 </r>
        <r>  0.1397  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000 </r>
       </set>
      </set>
      <set comment="kpoint 2">
       <set comment="band 1">
        <r>  0.9724  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000 </r>
        <r>  0.0005  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000 </r>
       </set>
       <set comment="band 2">
        <r>  0.0108  0.0002  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000 </r>
        <r>  0.1396  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000 </r>
        </set>
      </set>
     </set>
    </set>
   </array>
  </projected>
"""


class VaspXMLProjTestCase(NamedTuple):
    data: str
    id: str
    n_spins: int


proj_test_cases = [
    VaspXMLProjTestCase(
        data=NON_SPIN_POLARIZED_PROJ_ELEMENT, id="non_spin_polarized_proj", n_spins=1
    ),
    VaspXMLProjTestCase(data=SPIN_POLARIZED_PROJ_ELEMENT, id="spin_polarized_proj", n_spins=2),
    VaspXMLProjTestCase(data=NON_COLINEAR_PROJ_ELEMENT, id="non_colinear_proj", n_spins=4),
]


class TestVasprunProj:
    @pytest.fixture(params=proj_test_cases, ids=lambda c: c.id)  # pyright: ignore[reportUnknownLambdaType, reportUnknownMemberType]
    def case(self, request: pytest.FixtureRequest) -> VaspXMLProjTestCase:
        return request.param

    def test_proj_shape(self, case: VaspXMLProjTestCase) -> None:
        vaspxml = VaspXML.from_str(case.data)
        projected = vaspxml.projected
        assert projected is not None
        assert projected.shape == (2, 2, case.n_spins, 2, 9)


FINAL_STRUCTURE_ELEMENT = """ <structure name="finalpos" >
  <crystal>
   <varray name="basis" >
    <v>       3.84652000       0.00000000       0.00000000 </v>
    <v>       0.00000000       3.84652000       0.00000000 </v>
    <v>       0.00000000       0.00000000       3.84652000 </v>
   </varray>
   <i name="volume">     56.91201793 </i>
   <varray name="rec_basis" >
    <v>       0.25997525       0.00000000       0.00000000 </v>
    <v>       0.00000000       0.25997525       0.00000000 </v>
    <v>       0.00000000       0.00000000       0.25997525 </v>
   </varray>
  </crystal>
  <varray name="positions" >
   <v>       0.00000000       0.00000000       0.00000000 </v>
   <v>       0.50000000       0.50000000       0.50000000 </v>
   <v>       0.50000000       0.50000000       0.00000000 </v>
   <v>       0.50000000       0.00000000       0.50000000 </v>
   <v>       0.00000000       0.50000000       0.50000000 </v>
  </varray>
 </structure>
"""


def parse_final_structure_element() -> None:
    _parser = VaspXML.from_str(FINAL_STRUCTURE_ELEMENT)
    assert False
