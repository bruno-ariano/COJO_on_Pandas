###
# Bruno Ariano; bruno.ariano.87@gmail.com
# Join analysis using GCTA COJO concept
# This program is in pandas and need to be converted in pyspark

from pyspark.sql import SparkSession
import pyspark.sql.functions as f
import pyspark.sql.types as t
from pyspark import SparkConf
import numpy as np
import scipy.linalg
import pandas as pd
import scipy.stats
import os
import subprocess as sp

  
def find_median(values_list):
  median = np.median(values_list)
  return round(float(median),6)

med_find=f.udf(find_median,t.FloatType())

#########
# This function is used only for calculating the phenotype variance and the effective population size(though in general pops in GWAS are panmitic).
# Here I assume this is constant at 1.3. The real values looks like are not
# distant from this default.

def phenotype_and_neff_compute(data_ref):
  #Note: Below equation 8 in the cojo paper, is how we estimate the trait variance. \cr
  #   y'y = D_j S_j^2 (n-1) + D_j B_j^2 \cr
  #   Trait variance taken as the median of the equation above
  
  vars=data_ref["var_af"] *(data_ref["n_total"]) * data_ref["se"]**2  +  data_ref["var_af"] * (data_ref["n_total"]) * data_ref["beta"]**2
  vars = (find_median(vars/(data_ref["n_total"]-1)))
  #Here is stored the function to calculate the phenotypic variance. 
  # In my case I assume this to be constant at 1.3 given previous calculation
  # I have made and what I have observed in other repositories
  data_ref["neff"] = (vars - data_ref["var_af"] *data_ref["beta"]**2)/(data_ref["var_af"] *data_ref["se"]**2)
  return(vars,data_ref)

#Select SNP with the lowest conditioned p-value
def select_best_SNP(df, variants_conditioned):
  key_diff = set(df.SNP).difference(variants_conditioned.SNP)
  where_diff = df.SNP.isin(key_diff)
  df = df[where_diff]
  df_lowestp = df.loc[df["pval_cond"] == np.min(df["pval_cond"])]
  
  #if multiple variants are present with same p-value I select the first one
  if(df_lowestp.shape[0]>1):
      df_lowestp = df_lowestp.head(1)

  best_SNP_pos = df_lowestp["pos"]
  best_SNP_ID = df_lowestp["SNP"]
  best_SNP_pvalue = df_lowestp["pval_cond"]
  best_SNP_pos = int(best_SNP_pos)
  variants_conditioned = pd.concat([variants_conditioned,df_lowestp])
  return [best_SNP_ID, best_SNP_pos, best_SNP_pvalue, variants_conditioned]


### This follow the guideline reported here (https://www.mv.helsinki.fi/home/mjxpirin/GWAS_course/material/GWAS9.pdf)
def join_sumstat(SNP, var_af_slct, LD_matrix, betas, N, SE):
  b = betas*var_af_slct
  #This equation is reported here (https://www.karger.com/Article/FullText/513303#ref4) as a way to compute the OLS.
  ls_sumstat = scipy.linalg.solve(LD_matrix, b) #computes scaled lambdas as R^-1 * b.s
  sigma2_J_sumstat = find_median((b**2) + (N*var_af_slct*2*SE**2)) - np.dot(b.T, ls_sumstat)
  if np.any((sigma2_J_sumstat/N * np.diag(scipy.linalg.inv(LD_matrix))) <0):
    return "negative variance"
  ls_se_sumstat = np.sqrt(sigma2_J_sumstat/N * np.diag(scipy.linalg.inv(LD_matrix))) #SEs of scaled lambdas 
  ls_sumstat = (ls_sumstat/var_af_slct)
  ls_se_sumstat = (ls_se_sumstat/var_af_slct)
  
  new_z = (np.abs(ls_sumstat/ls_se_sumstat))
  new_pval = scipy.stats.norm.sf(abs(new_z))*2
  d = {"SNP": SNP, "beta_cond_tmp": ls_sumstat, "se_cond_tmp" : ls_se_sumstat, "pval_cond_tmp":new_pval}
  res_data = pd.DataFrame.from_dict(d)
  return res_data[res_data["SNP"]==SNP[0]]


# The only measures that needs to be calculated is the the 
# phenotype variance var_y if is not considered constant.
p_value_threshold = 5e-8
max_iter = 100

#establish spark connection
spark = SparkSession.builder.config("spark.some.config.option", "some-value").master("local").getOrCreate()

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
          .select("SNP","beta","MAF_sumstat","se","pval","n_total")
)

### Consider only SNP that overlap between the summary stat and the reference
bim_uk_freq_filtered_SNP = bim_uk_freq.join(sumstat, on = ["SNP"], how = "inner")

### Various filtering and calculation of parameters


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
                        .select("SNP" ,"pos","MAF","beta", "beta_cond", "se", "se_cond", "pval", "pval_cond", "n_total")
                        .withColumn("var_af", (2*f.col("MAF")*(1-f.col("MAF"))))
                        .toPandas())


#### At this point I need to create a matrix with the LD information
                            
bim_uk_freq_filtered_SNP["SNP"].to_csv("test_data_single_window/variants_plink.csv", sep = " ", header = False)

input_bfile = "test_data_single_window/ukb_v3_chr1.downsampled10k_window"
out_bfile= "test_data_single_window/ukb_v3_chr1.downsampled10k_window_joint"
variants = "test_data_single_window/variants_plink.csv"

cmd_bim_create = [
        'plink',
        '--bfile', input_bfile,
        '--extract',  variants,
        '--allow-extra-chr',
        '--make-bed',
        '--out', out_bfile
    ]

cmd_str = ' '.join([str(x) for x in cmd_bim_create])

sp.call(cmd_str, shell=True)

SNP_name_LD = pd.read_csv("test_data_single_window/ukb_v3_chr1.downsampled10k_window_joint.bim", sep = "\t", names = ["chr","SNP","rec","pos","A1","A2"])

bfile = "test_data_single_window/ukb_v3_chr1.downsampled10k_window_joint"
out_ld = "test_data_single_window/LD_SNP_test"

cmd_ld_create = [
        'plink',
        '--bfile', bfile,
        '--out', out_ld,
        '--r2 square'
    ]
cmd_str = ' '.join([str(x) for x in cmd_ld_create])

sp.call(cmd_str, shell=True)

ld_matrix = np.loadtxt('test_data_single_window/LD_SNP_test.ld')
ld_matrix_names = pd.DataFrame(ld_matrix, index=SNP_name_LD["SNP"], columns=SNP_name_LD["SNP"])

# I select the SNP with the lowest p-value making sure that it wasn't selected previously.

### Technically this is the first cycle of the analysis

variants_conditioned = pd.DataFrame({'SNP' : []})
best_SNP_ID, best_SNP_pos, best_SNP_pvalue, variants_conditioned =  select_best_SNP(bim_uk_freq_filtered_SNP, variants_conditioned)

best_SNPs_cond = np.empty((0,1), str)
best_SNPs_cond = np.append(best_SNPs_cond, best_SNP_ID)


# In this particular script the LD window is created only once and used as a reference for each iteration
# When the whole summary stat is taken then a bigger LD matrix need to be consulted using a Hail table

#Either the minimum p-value is reached or the iteration reach a maximum
iters = 0
while(np.any(best_SNP_pvalue<p_value_threshold) and iters < max_iter):
    for index, row in bim_uk_freq_filtered_SNP.iterrows():
                            
      #Skip if the SNP to condition is included in the conditioned list
      if np.any(variants_conditioned["SNP"] == row["SNP"]):
        continue
      best_SNPs_cond_tmp = np.append(row["SNP"], best_SNPs_cond)
      select_rows = bim_uk_freq_filtered_SNP.loc[bim_uk_freq_filtered_SNP["SNP"].isin(best_SNPs_cond_tmp)]
      betas_slct = np.array(select_rows["beta_cond"])
      var_af_slct = np.array(select_rows["var_af"])               
      SE_select = np.array(select_rows["se_cond"])
      N_slct = np.array(select_rows["n_total"])
      ld_matrix_slct_col = ld_matrix_names[best_SNPs_cond_tmp]
      ld_matrix_slct = ld_matrix_slct_col.loc[ld_matrix_slct_col.index.isin(best_SNPs_cond_tmp)].values
      out_col = ld_matrix_names[best_SNPs_cond_tmp]
      out = out_col.loc[out_col.index.isin(best_SNPs_cond_tmp)].values

      np.fill_diagonal(out,0) 
      
      # if the SNPs considered has a R2 higher than 0.9 with any of the top variants its p-value is set to 1
      
      if np.any(out>0.9):
          #bim_uk_freq_filtered_SNP = bim_uk_freq_filtered_SNP.loc[bim_uk_freq_filtered_SNP['SNP'] != row["SNP"]]
          bim_uk_freq_filtered_SNP.loc[bim_uk_freq_filtered_SNP["SNP"] == row["SNP"],"pval_cond"] = 1
          continue
      
      #sum_stat_filtered_SNP_tmp = join_sumstat_sojo(best_SNPs_cond_tmp, MAF_select, D_neff_slct, ld_matrix_slct, betas_slct, N_slct, SE_select, var_y)
      #Exclude if we have a negative 
      sum_stat_filtered_SNP_tmp = join_sumstat(best_SNPs_cond_tmp, var_af_slct, ld_matrix_slct,betas_slct, N_slct, SE_select)
      if np.any(sum_stat_filtered_SNP_tmp == "collinearity"):
        bim_uk_freq_filtered_SNP = bim_uk_freq_filtered_SNP.loc[bim_uk_freq_filtered_SNP['SNP'] != row["SNP"]]
        continue
                            
      bim_uk_freq_filtered_SNP.loc[bim_uk_freq_filtered_SNP["SNP"] == row["SNP"],"beta_cond"] = sum_stat_filtered_SNP_tmp["beta_cond_tmp"][0]
      bim_uk_freq_filtered_SNP.loc[bim_uk_freq_filtered_SNP["SNP"] == row["SNP"],"se_cond"] = sum_stat_filtered_SNP_tmp["se_cond_tmp"][0]
      bim_uk_freq_filtered_SNP.loc[bim_uk_freq_filtered_SNP["SNP"] == row["SNP"],"pval_cond"] = sum_stat_filtered_SNP_tmp["pval_cond_tmp"][0]

    print("end of cycle")
    
    #Here the selection of the best SNP is made again and the cycle is repeated
    best_SNP_ID, best_SNP_pos, best_SNP_pvalue, variants_conditioned =  select_best_SNP(bim_uk_freq_filtered_SNP, variants_conditioned)
    best_SNPs_cond = np.append(best_SNP_ID, best_SNPs_cond)
    iters = iters+1
    print(iters)

bim_uk_freq_filtered_SNP_final = bim_uk_freq_filtered_SNP.loc[bim_uk_freq_filtered_SNP["pval_cond"]<5e-08]
bim_uk_freq_filtered_SNP_final = bim_uk_freq_filtered_SNP_final.loc[bim_uk_freq_filtered_SNP_final["pval_cond"] != 1]

#bim_uk_freq_filtered_SNP_final.repartition(1).write.format("csv").save("test_data_single_window/cojo_out/GCST002222_cojo_on_spark")
