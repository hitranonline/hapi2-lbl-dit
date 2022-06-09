from time import time
from warnings import warn

#from .discrete_integral_transform_hapi import synthesize_spectrum
from .discrete_integral_transform_hapi import synthesize_spectrum, SETTINGS

from hapi import calculateProfileParametersVoigt, \
    save_to_file, listOfTuples, getDefaultValuesForXsect, \
    arange_, volumeConcentration, ISO, ISO_INDEX, \
    LOCAL_TABLE_CACHE, PYTIPS, DefaultIntensityThreshold, \
    DefaultOmegaWingHW, CaselessDict, calculate_parameter_Sw

from numpy import zeros, array

__FloatType__ = float

VARIABLES = {}
VARIABLES['abscoef_debug'] = True

ABSCOEF_DOCSTRING_TEMPLATE = """
    INPUT PARAMETERS: 
        Components:  list of tuples [(M,I,D)], where
                        M - HITRAN molecule number,
                        I - HITRAN isotopologue number,
                        D - relative abundance (optional)
        SourceTables:  list of tables from which to calculate cross-section   (optional)
        partitionFunction:  pointer to partition function (default is PYTIPS) (optional)
        Environment:  dictionary containing thermodynamic parameters.
                        'p' - pressure in atmospheres,
                        'T' - temperature in Kelvin
                        Default={{'p':1.,'T':296.}}
        WavenumberRange:  wavenumber range to consider.
        WavenumberStep:   wavenumber step to consider. 
        WavenumberWing:   absolute wing for calculating a lineshape (in cm-1) 
        WavenumberWingHW:  relative wing for calculating a lineshape (in halfwidths)
        IntensityThreshold:  threshold for intensities
        Diluent:  specifies broadening mixture composition, e.g. {{'air':0.7,'self':0.3}}
        HITRAN_units:  use cm2/molecule (True) or cm-1 (False) for absorption coefficient
        File:   write output to file (if specified)
        Format:  c-format of file output (accounts for significant digits in WavenumberStep)
        LineMixingRosen: include 1st order line mixing to calculation
    OUTPUT PARAMETERS: 
        Wavenum: wavenumber grid with respect to parameters WavenumberRange and WavenumberStep
        Xsect: absorption coefficient calculated on the grid
    ---
    DESCRIPTION:
        Calculate absorption coefficient using {profile}.
        Absorption coefficient is calculated at arbitrary temperature and pressure.
        User can vary a wide range of parameters to control a process of calculation.
        The choise of these parameters depends on properties of a particular linelist.
        Default values are a sort of guess which gives a decent precision (on average) 
        for a reasonable amount of cpu time. To increase calculation accuracy,
        user should use a trial and error method.
    ---
    EXAMPLE OF USAGE:
        {usage_example}
    ---
    """     

def absorptionCoefficient_Voigt(Components=None,SourceTables=None,partitionFunction=PYTIPS,
                                  Environment=None,OmegaRange=None,OmegaStep=None,OmegaWing=None,
                                  IntensityThreshold=DefaultIntensityThreshold,
                                  OmegaWingHW=DefaultOmegaWingHW,
                                  GammaL='gamma_air', HITRAN_units=True, LineShift=True,
                                  File=None, Format=None, OmegaGrid=None,
                                  WavenumberRange=None,WavenumberStep=None,WavenumberWing=None,
                                  WavenumberWingHW=None,WavenumberGrid=None,
                                  Diluent={},LineMixingRosen=False,
                                  profile=None,calcpars=None,exclude=set(),
                                  DEBUG=None):
                                                              
    if DEBUG is not None: 
        VARIABLES['abscoef_debug'] = True
    else:
        VARIABLES['abscoef_debug'] = False
        
    if not LineMixingRosen: exclude.add('YRosen')
    if not LineShift: exclude.update({'Delta0','Delta2'})
    
    # Parameters OmegaRange,OmegaStep,OmegaWing,OmegaWingHW, and OmegaGrid
    # are deprecated and given for backward compatibility with the older versions.
    if WavenumberRange is not None:  OmegaRange=WavenumberRange
    if WavenumberStep is not None:   OmegaStep=WavenumberStep
    if WavenumberWing is not None:   OmegaWing=WavenumberWing
    if WavenumberWingHW is not None: OmegaWingHW=WavenumberWingHW
    if WavenumberGrid is not None:   OmegaGrid=WavenumberGrid

    if OmegaWing is not None: warn('WavenumberWing/OmegaWing parameter is not considered in this implementation')
    if OmegaWingHW is not None: warn('WavenumberWingHW/OmegaWingHW parameter is not considered in this implementation')

    # "bug" with 1-element list
    Components = listOfTuples(Components)
    SourceTables = listOfTuples(SourceTables)
    
    # determine final input values
    Components,SourceTables,Environment,OmegaRange,OmegaStep,OmegaWing,\
    IntensityThreshold,Format = \
       getDefaultValuesForXsect(Components,SourceTables,Environment,OmegaRange,
                                OmegaStep,OmegaWing,IntensityThreshold,Format)
    
    if OmegaStep>0.1: 
        warn('Big wavenumber step: possible accuracy decline')

    # get uniform linespace for cross-section
    #number_of_points = (OmegaRange[1]-OmegaRange[0])/OmegaStep + 1
    #Omegas = linspace(OmegaRange[0],OmegaRange[1],number_of_points)
    if OmegaGrid is not None:
        #Omegas = npsort(OmegaGrid)
        Omegas = OmegaGrid
    else:
        #Omegas = arange(OmegaRange[0],OmegaRange[1],OmegaStep)
        Omegas = arange_(OmegaRange[0],OmegaRange[1],OmegaStep) # fix
    number_of_points = len(Omegas)
    Xsect = zeros(number_of_points)
       
    # reference temperature and pressure
    T_ref_default = __FloatType__(296.) # K
    p_ref_default = __FloatType__(1.) # atm
    
    # actual temperature and pressure
    T = Environment['T'] # K
    p = Environment['p'] # atm
       
    # create dictionary from Components
    ABUNDANCES = {}
    NATURAL_ABUNDANCES = {}
    for Component in Components:
        M = Component[0]
        I = Component[1]
        if len(Component) >= 3:
            ni = Component[2]
        else:
            try:
                ni = ISO[(M,I)][ISO_INDEX['abundance']]
            except KeyError:
                raise Exception('cannot find component M,I = %d,%d.' % (M,I))
        ABUNDANCES[(M,I)] = ni
        NATURAL_ABUNDANCES[(M,I)] = ISO[(M,I)][ISO_INDEX['abundance']]
        
    # pre-calculation of volume concentration
    if HITRAN_units:
        factor = __FloatType__(1.0)
    else:
        factor = volumeConcentration(p,T)
        
    # setup the Diluent variable
    GammaL = GammaL.lower()
    if not Diluent:
        if GammaL == 'gamma_air':
            Diluent = {'air':1.}
        elif GammaL == 'gamma_self':
            Diluent = {'self':1.}
        else:
            raise Exception('Unknown GammaL value: %s' % GammaL)
        
    # Simple check
    print(Diluent)  # Added print statement # CHANGED RJH 23MAR18  # Simple check
    for key in Diluent:
        val = Diluent[key]
        if val < 0 or val > 1: # if val < 0 and val > 1:# CHANGED RJH 23MAR18
            raise Exception('Diluent fraction must be in [0,1]')
            
    # ================= CALCULATE ARRAYS OF PARAMETERS =====================

    t = time()
    
    CALC_INFO_TOTAL = []
    
    CALC_PARAMS = {}
    
    PARNAMES_INPUT = ['Nu','GammaD','Gamma0','Sw','Delta0']
    for parname in PARNAMES_INPUT:
        CALC_PARAMS[parname] = []
    
    # SourceTables contain multiple tables
    for TableName in SourceTables:
    
        # exclude parameters not involved in calculation
        DATA_DICT = LOCAL_TABLE_CACHE[TableName]['data']
        parnames_exclude = ['a','global_upper_quanta','global_lower_quanta',
            'local_upper_quanta','local_lower_quanta','ierr','iref','line_mixing_flag'] 
        parnames = set(DATA_DICT)-set(parnames_exclude)
        
        nlines = len(DATA_DICT['nu'])

        for RowID in range(nlines):
                            
            # create the transition object
            TRANS = CaselessDict({parname:DATA_DICT[parname][RowID] for parname in parnames}) # CORRECTLY HANDLES DIFFERENT SPELLING OF PARNAMES
            TRANS['T'] = T
            TRANS['p'] = p
            TRANS['T_ref'] = T_ref_default
            TRANS['p_ref'] = p_ref_default
            TRANS['Diluent'] = Diluent
            TRANS['Abundances'] = ABUNDANCES
            
            # filter by molecule and isotopologue
            if (TRANS['molec_id'],TRANS['local_iso_id']) not in ABUNDANCES: continue
                
            #   FILTER by LineIntensity: compare it with IntencityThreshold
            TRANS['SigmaT']     = partitionFunction(TRANS['molec_id'],TRANS['local_iso_id'],TRANS['T'])
            TRANS['SigmaT_ref'] = partitionFunction(TRANS['molec_id'],TRANS['local_iso_id'],TRANS['T_ref'])
            LineIntensity = calculate_parameter_Sw(None,TRANS)
            if LineIntensity < IntensityThreshold: continue

            # calculate profile parameters 
            if VARIABLES['abscoef_debug']:
                CALC_INFO = {}
            else:
                CALC_INFO = None                
            PARAMETERS = calculateProfileParametersVoigt(TRANS=TRANS,CALC_INFO=CALC_INFO,exclude=exclude)
                               
            # append parameters to the arrays
            for parname in PARNAMES_INPUT:
                CALC_PARAMS[parname].append(PARAMETERS[parname])
                               
            # append debug information for the abscoef routine                
            if VARIABLES['abscoef_debug']: DEBUG.append(CALC_INFO)
        
    print('%f seconds elapsed for preparing parameters; nlines = %d'%(time()-t,nlines))
    
    # ================= CALCULATE ABSCOEF VIA INTEGRAL TRANSFORMATION =====================
    
    for parname in PARNAMES_INPUT:
        CALC_PARAMS[parname] = array(CALC_PARAMS[parname]) 
        
    print('starting synthesize_spectrum...')
        
    dxG = 0.1
    dxL = 0.1
    folding_thresh = 1e-6

    # Nu,GammaD,Gamma0,Delta0,WnGrid,YRosen=0.0,Sw=1.0 => PROFILE_VOIGT parameter names

    t = time()    
    I0_lin, S_klm_lin = synthesize_spectrum(
        v_ax=Omegas,
        v0=CALC_PARAMS['Nu']+CALC_PARAMS['Delta0'],
        wG=2*CALC_PARAMS['GammaD'],
        wL=2*CALC_PARAMS['Gamma0'],
        S0=CALC_PARAMS['Sw'],
        dxG=dxG,dxL=dxL,
        folding_thresh=folding_thresh
    )
    print('%f seconds elapsed for synthesize_spectrum'%(time()-t,))
    
    Xsect = I0_lin
    
    if File: save_to_file(File,Format,Omegas,Xsect)
    return Omegas,Xsect    
    
absorptionCoefficient_Voigt.__doc__ = ABSCOEF_DOCSTRING_TEMPLATE.format(
    profile='Voigt',
    usage_example="""
        nu,coef = absorptionCoefficient_Voigt(((2,1),),'co2',WavenumberStep=0.01,
                                              HITRAN_units=False,Diluent={'air':1})
    """
)
    