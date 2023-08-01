from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import StandardScaler


class Features(object):
    def __init__(self):
        # reference
        self.reference = ['病歷號']


        # Gender data (categorical)
        self.gender_data = ['Gender']
        
        
        # Age data (numerical)
        self.age_data = ['Age']


        # blood data (numerical)
        # WBC: white blood cell count
        # Hb: 血色素
        # Plt: 血小板
        self.blood_data = ['WBC', 'Hb', 'Plt', 'PB_Blast']
        self.WBC_data = ['WBC']


        # Karyotype data
        self.karyotype_prenamed = ['Karyotype ']
        self.karyotype = [element.strip() for element in self.karyotype_prenamed]
        
        
        # ELN 2022 Table 6: https://ashpublications.org/blood/article/140/12/1345/485817/Diagnosis-and-management-of-AML-in-adults-2022
        # Refinement of cytogenetic classification in acute myeloid leukemia: https://ashpublications.org/blood/article/116/3/354/27618/Refinement-of-cytogenetic-classification-in-acute
        self.karyotype_data = ['t(8;21)', 'inv(16)', 't(16;16)',                                                      # favorable
                                   't(9;11)',                                                                             # intermediate
                                   't(6;9)', 't(9;22)', 't(8;16)', 'inv(3)', 't(3;3)', '-5', 'del(5q)', '-7', '-17']      # adverse: t(v;11q23.3), t(3q26.2;v), abn(17p)
        
        
        # gene contain pathogenic variant labeled data (categorical)
        self.mutation_data = ['PTPN11', 'NRAS', 'KRAS', 'NPM1', 'FLT_TKD', 'MLL', 'KIT', 'RUNX1', 'WT1', 'ASXL1', 'IDH1',
                              'IDH2', 'TET2', 'DNMT3A', 'TP53', 'SF3B1', 'U2AF1', 'SRSF2', 'ZRSR2', 'GATA2', 'STAG1', 
                              'STAG2', 'Rad21', 'SMC1A', 'SMC3', 'PHF6', 'CBL-b', 'c-CBL', 'ETV6', 'EZH2', 'BCOR']
        
        
        # label data (numerical)
        # EFS is missing
        self.label_coding_data = ['Standard', 'RFS_coding_no_censor_2021', 'OS_coding_no_censor_2021',]
        self.label_time_data = ['RFS_no_censor_2021', 'OSD_no_censor_2021']
        self.label_data = ['RFS_label', 'OS_label']


        # comparison 
        self.comparision_data = ['ELN_2017', 'Revised_ELN2022_Tsai', 'HSCT']


        # preprocessing pipeline:
        age_preprocessing = Pipeline([
            ('scaler_standard', StandardScaler())
        ])
        blood_preprocessing = Pipeline([
            ('imputer_bayesian_ridge', IterativeImputer(random_state=0)),
            ('scaler_standard', StandardScaler())
        ])  
        self.preprocessor = ColumnTransformer([
            ('age_preprocessing', age_preprocessing, self.age_data),
            ('blood_preprocessing', blood_preprocessing, self.blood_data),
        ])





