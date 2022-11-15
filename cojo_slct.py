###
# Bruno Ariano; bruno.ariano.87@gmail.com
# Join analysis using GCTA COJO concept
# This program is in pandas and need to be converted in pyspark

from pyspark.sql import SparkSession
import pyspark.sql.functions as f
import pyspark.sql.types as t
from pyspark import SparkConf
from pyspark.context import SparkContext as c 
import numpy as np
import scipy.linalg
import pandas as pd
import os
import subprocess as sp
import dask.array as da
from statsmodels.stats.outliers_influence import variance_inflation_factor
from functools import reduce
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


##########
# This function is for dealing with multicollinearity problems.  
# The function it self take long type for computation. 

def calc_VIF(x):
  vif= pd.DataFrame()
  vif["VIF"]=[variance_inflation_factor(x.values,i) for i in range(x.shape[1])]
  vif['variables']=x.columns
  
def find_median(values_list):
  median = np.median(values_list)
  return round(float(median),6)

med_find=f.udf(find_median,t.FloatType())


# This function is used only for calculating the phenotype variance and the effective population size(though in general pops in GWAS are panmitic).
# Here I assume this is constant at 1.3. The real values looks like are not
# distant from this default.

def phenotype_and_neff_compute(data_ref):
  # Note: Below equation 8 in the cojo paper, is how we estimate the trait variance. \cr
  # y'y = D_j S_j^2 (n-1) + D_j B_j^2 \cr
  # Trait variance taken as the median of the equation above
  vars= (data_ref["var_af"] *(data_ref["n_total"]) * data_ref["se"]**2  +  data_ref["var_af"] * data_ref["beta"]**2)
  vars = find_median(vars)
  data_ref["neff"] = (vars/(data_ref["var_af"] *data_ref["se"]**2)) - (((data_ref["beta"]**2))/data_ref["se"]**2)
  return(vars,data_ref)


# This function find the SNP with the lowest p-value in a dataset.
# Given that the dataset is sorted before I just take the first row
# after filtering for SNPs already conditioned

def select_best_SNP(df, variants_conditioned):
  key_diff = set(df.SNP).difference(variants_conditioned.SNP)
  where_diff = df.SNP.isin(key_diff)
  df = df[where_diff]
  df_lowestp = df.loc[df["pval_cond"] == np.min(df["pval_cond"])]
  if(df_lowestp.shape[0]>1):
      df_lowestp = df_lowestp.head(1)
  best_SNP_pos = df_lowestp["pos"]
  best_SNP_ID = df_lowestp["SNP"]
  best_SNP_pvalue = df_lowestp["pval_cond"]
  best_SNP_pos = int(best_SNP_pos)
  #best_SNP_conditioned = df.filter(f.col("pval") == best_SNP_pvalue).select("SNP")
  #variants_conditioned = variants_conditioned.union(best_SNP_conditioned)
  variants_conditioned = pd.concat([variants_conditioned,df_lowestp])
  return [best_SNP_ID, best_SNP_pos, best_SNP_pvalue, variants_conditioned]


#This version of the function is the same adopted in the gcta cojo software
def join_sumstat_gcta(SNP,var_af, LD_matrix, var_y, betas,N_eff):
  # Here I assume Dw * WW' * Dw explained in the paper
  # can be approximated by the LD correlation following what published here (insert link)
  #B = np.dot(np.dot(np.sqrt(np.diag(var_af).astype(float)), LD_matrix.astype(float)), np.sqrt(np.diag(var_af).astype(float)))
  #new_betas = np.dot(np.dot(np.linalg.inv(B * N_eff), np.diag(var_af  * min(N_eff))), betas)
  N_eff_min = min(N_eff)
  B = np.dot(np.dot(np.sqrt(np.diag(var_af).astype(float)), LD_matrix.astype(float)), np.sqrt(np.diag(var_af).astype(float)))
  for j in range(B.shape[1]):
    for k in range(j,B.shape[1]):
      B[j,k] = min(N_eff[j], N_eff[k]) *  B[j,k]
      if k!=j :
          B[k,j] = min(N_eff[j], N_eff[k]) * B[k,j]
         
  new_betas = np.dot(np.linalg.inv(B), np.dot(np.diag(var_af * N_eff), betas))
  
  #Alternative way for calculating joint se
  
  #sigma_joint = np.abs((var_y - np.dot(var_af , new_betas**2))/(len(SNP)-1))
  #new_se = np.sqrt(np.diag(((sigma_joint) * np.linalg.inv(B))/(N_eff-2)))
  
  
  new_se = np.sqrt(np.diag(((var_y) * np.linalg.inv(B))))
  new_z = (np.abs(new_betas/new_se))
  new_pval = scipy.stats.chi2.sf((new_z**2), len(SNP))
  d = {"SNP": SNP, "beta_cond_tmp": new_betas, "se_cond_tmp" : new_se, "pval_cond_tmp":new_pval }
  res_data = pd.DataFrame.from_dict(d)
  return res_data[res_data["SNP"]==SNP[0]]

p_value_threshold = 5e-8
max_iter = 100

#establish spark connection
spark = SparkSession.builder.config("some_config", "some_value").master("local[*]").getOrCreate()

### load reference
schema_bim = t.StructType([ 
    t.StructField("CHR",t.StringType(),True), 
    t.StructField("SNP",t.StringType(),True), 
    t.StructField("rec_rate",t.StringType(),True), 
    t.StructField("POS",t.StringType(),True), 
    t.StructField("A1",t.StringType(),True), 
    t.StructField("A2",t.StringType(),True), 
  ])

schema_freq = t.StructType([ 
    t.StructField("CHR",t.StringType(),True), 
    t.StructField("SNP",t.StringType(),True), 
    t.StructField("A1",t.StringType(),True), 
    t.StructField("A2",t.StringType(),True), 
    t.StructField("MAF",t.StringType(),True), 
    t.StructField("POS",t.StringType(),True), 
  ])

bim_uk = spark.read.format("csv").load("test_data_single_window/ukb_v3_chr1.downsampled10k_window.reshaped.bim", sep = ",", schema = schema_bim)

# The allele frequencies are pre-calculated using plink on the entire reference chromosome.
freq_uk = spark.read.format("csv").load("test_data/ukb_v3_chr1.downsampled10k.reshaped.frq", sep = ",", schema = schema_freq)
freq_uk = freq_uk.select("SNP", "MAF")
bim_uk_freq = bim_uk.join(freq_uk, on = ["SNP"])

### load sumstat
sumstat = spark.read.format("parquet").load("test_data/GCST002222.parquet")
sumstat = (sumstat
          .withColumn("SNP", f.concat_ws(":", f.col("chrom"), f.col("pos"), f.col("ref"), f.col("alt")))
          .withColumn("MAF_sumstat", f.when(f.col("eaf")>0.5, 1-f.col("eaf")).otherwise(f.col("eaf")))
          .select("SNP","beta","MAF_sumstat","se","pval","n_total", "eaf", "ref", "alt")
)

### Consider only SNP that overlap between the summary stat and the reference
bim_uk_freq_filtered_SNP = bim_uk_freq.join(sumstat, on = ["SNP"], how = "inner")

### Various filtering and calculation of parameters

# Calculate the allele frequency variance
bim_uk_freq_filtered_SNP = (bim_uk_freq_filtered_SNP
                            .withColumn("var_af", (2*f.col("eaf")*(1-f.col("eaf")))))


# Filter for SNPs that have no more than 0.2 difference in the 
# allele frequencies between the reference anf the summary stat.
# I also filter out variants with allele frequencies lower than 1%

bim_uk_freq_filtered_SNP = (bim_uk_freq_filtered_SNP
                        .withColumn("beta_cond", f.col("beta"))
                        .withColumn("pval_cond", f.col("pval"))
                        .withColumn("se_cond", f.col("se"))
                        .withColumn("diff_af", f.abs(f.col("MAF_sumstat")-f.col("MAF")))
                        .filter(f.col("diff_af")<0.2)
                        .filter(f.col("MAF")>0.01)
                        .select("SNP" ,"pos","var_af","MAF" ,"beta", "beta_cond", "se", "se_cond", "pval", "pval_cond", "n_total", "eaf",  "ref", "alt")
                        .toPandas())


schema_SNP_bim = t.StructType([
    t.StructField("chr", t.IntegerType(), True),
    t.StructField("SNP", t.StringType(), True),
    t.StructField("rec", t.IntegerType(), True),
    t.StructField("pos", t.IntegerType(), True),
    t.StructField("A1", t.StringType(), True),
    t.StructField("A2", t.StringType(), True)])



#load the LD matrix
ld_matrix = np.loadtxt('/Users/ba13/Desktop/Open_Target_Genetics/creation_pipeline/COJO_on_SPARK/test_data_single_window/LD_SNP_test.ld')
ld_matrix_names = pd.DataFrame(ld_matrix, index=SNP_name_LD["SNP"], columns=SNP_name_LD["SNP"])

# switch the sign of LD if is the minor in the allele frequency bim file
SNP_name_LD = bim_uk.withColumn("SNP_bim", f.concat_ws(":", f.col("chr"), f.col("pos"), f.col("A1"), f.col("A2")))
SNP_switch = SNP_name_LD.filter(f.col("SNP") != f.col("SNP_bim")).select("SNP")
SNP_switch = SNP_switch.toPandas()
SNP_name_LD = SNP_name_LD.toPandas()

ld_matrix_names[SNP_switch["SNP"]] = ld_matrix_names[SNP_switch["SNP"]]*-1
ld_matrix_names.loc[SNP_switch["SNP"]] = ld_matrix_names.loc[SNP_switch["SNP"]]*-1


#Here I calculate the pehontypic variance and the sample size
var_y, bim_uk_freq_filtered_SNP = phenotype_and_neff_compute(bim_uk_freq_filtered_SNP)

# I select the SNP with the lowest p-value making sure that it wasn't selected previously.
variants_conditioned = pd.DataFrame({'SNP' : []})
best_SNP_ID, best_SNP_pos, best_SNP_pvalue, variants_conditioned =  select_best_SNP(bim_uk_freq_filtered_SNP, variants_conditioned)

best_SNPs_cond = np.empty((0,1), str)
best_SNPs_cond = np.append(best_SNPs_cond, best_SNP_ID)



iters = 0
f=0
while(np.any(best_SNP_pvalue<p_value_threshold) and iters < max_iter):
    print("ok")
    for index, row in bim_uk_freq_filtered_SNP.iterrows():
      
      #Skip if the SNP to condition is included in the conditioned list
      if np.any(variants_conditioned["SNP"] == row["SNP"]):
        continue
      #Select SNP to condition
      best_SNPs_cond_tmp = np.append(row["SNP"], best_SNPs_cond)
      select_rows = reduce(pd.DataFrame.append,(map(lambda i: bim_uk_freq_filtered_SNP[bim_uk_freq_filtered_SNP.SNP== i], best_SNPs_cond_tmp)))
      var_y, select_rows = phenotype_and_neff_compute(select_rows)
      betas_slct = np.array(select_rows["beta"])
      var_af_select = np.array(select_rows["var_af"])
      SE_select = np.array(select_rows["se"])
      N_slct = np.array(select_rows["neff"])
      ld_matrix_slct = ld_matrix_names[best_SNPs_cond_tmp].loc[best_SNPs_cond_tmp]
      out = ld_matrix_names[best_SNPs_cond_tmp].loc[best_SNPs_cond_tmp]
      np.fill_diagonal(out.values,0) 
      
      # if the SNPs considered has a R2 higher than 0.9 with any of the top variants I set the p-value to 1 (gcta suggestion)
      if (np.any((out**2)> 0.9):
          bim_uk_freq_filtered_SNP.loc[bim_uk_freq_filtered_SNP["SNP"] == row["SNP"],"pval_cond"] = 1
          continue
      
      sum_stat_filtered_SNP_tmp = join_sumstat_gcta(best_SNPs_cond_tmp, var_af_select, ld_matrix_slct,var_y, betas_slct, N_slct)
      bim_uk_freq_filtered_SNP.loc[bim_uk_freq_filtered_SNP["SNP"] == row["SNP"],"beta_cond"] = sum_stat_filtered_SNP_tmp["beta_cond_tmp"][0]
      bim_uk_freq_filtered_SNP.loc[bim_uk_freq_filtered_SNP["SNP"] == row["SNP"],"se_cond"] = sum_stat_filtered_SNP_tmp["se_cond_tmp"][0]
      bim_uk_freq_filtered_SNP.loc[bim_uk_freq_filtered_SNP["SNP"] == row["SNP"],"pval_cond"] = sum_stat_filtered_SNP_tmp["pval_cond_tmp"][0]
    
    print("end of cycle")
    #Here the selection of the best SNP is made again and the cycle is repeated
    best_SNP_ID, best_SNP_pos, best_SNP_pvalue, variants_conditioned =  select_best_SNP(bim_uk_freq_filtered_SNP, variants_conditioned)
    best_SNPs_cond = np.append(best_SNP_ID, best_SNPs_cond)
    iters = iters+1
    print(iters)

#Final list of SNPs conditioned
print(bim_uk_freq_filtered_SNP.loc[bim_uk_freq_filtered_SNP["pval_cond"]< 5e-8])

