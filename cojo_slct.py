###
# Bruno Ariano; bruno.ariano.87@gmail.com
# Join analysis using GCTA COJO concept; 2022
# This program is in pandas and need to be converted in pyspark

from pyspark.sql import SparkSession
import pyspark.sql.functions as f
import pyspark.sql.types as t
from pyspark import SparkConf
from pyspark.context import SparkContext as c 
import numpy as np
import scipy.linalg
import pandas as pd
import scipy.stats
import os
import subprocess as sp
import dask.array as da
from statsmodels.stats.outliers_influence import variance_inflation_factor


##########
# This function is for dealing with multicollinearity problems.  
# The function it self take long type for computation. 

def calc_VIF(x):
  vif= pd.DataFrame()
  vif["VIF"]=[variance_inflation_factor(x.values,i) for i in range(x.shape[1])]
  vif['variables']=x.columns

#########
# This function is used only for calculating the phenotype varaince.
# Here I assume this is constant at 1.3. The real values looks like are not
# distant from this default.

def phenotype_var_compute():
  #Here is stored the function to calculate the phenotypic variance. 
  # In my case I assume this to be constant at 1.3 given previous calculation
  # I have made and what I have observed in other repositories
  print("ok")
  
def find_median(values_list):
  median = np.median(values_list)
  return round(float(median),2)

med_find=f.udf(find_median,t.FloatType())


#########
# This function find the SNP with the lowest p-value in a dataset.
# Given that the dataset is sorted before I just take the first row
# after filtering for SNPs already conditioned

def select_best_SNP(df, variants_conditioned):
  key_diff = set(df.SNP).difference(variants_conditioned.SNP)
  where_diff = df.SNP.isin(key_diff)
  df = df[where_diff]
  df_lowestp = df.loc[df["pval"] == np.min(df["pval"])]
  #df = df.join(f.broadcast(variants_conditioned), on = ["SNP"], how = "left_anti")
  #R = df.head(1)[0]
  best_SNP_pos = df_lowestp["pos"]
  best_SNP_ID = df_lowestp["SNP"]
  best_SNP_pvalue = df_lowestp["pval"]
  best_SNP_pos = int(best_SNP_pos)
  #best_SNP_conditioned = df.filter(f.col("pval") == best_SNP_pvalue).select("SNP")
  #variants_conditioned = variants_conditioned.union(best_SNP_conditioned)
  variants_conditioned = pd.concat([variants_conditioned,df_lowestp])
  return [best_SNP_ID, best_SNP_pos, best_SNP_pvalue, variants_conditioned]


##########
# This function create a window of 4Mb (+/- 2Mb) around the selected variant
# Moreover it write the sets of variant that will later be used by plink to 
# calculate the LD matrix

#def create_windows_plink(df, best_SNP_pos):
#  df["pos"] = df["pos"].astype(int)
#  df_wind = df.loc[(df["pos"] > best_SNP_pos - 2e6) &  (df["pos"] < best_SNP_pos + 2e6)]
#  df_wind_SNP = df_wind["SNP"]
#  return (df_wind)
#  df_wind_SNP.to_csv("test_data_single_window/variants_plink.csv", sep = " ", header = False)


#########
# This is the first verion of the joint analysis.
# This version follow mostly what's in the paper and also indicated in the coco(https://github.com/theboocock/coco/blob/master/R/joint_algebra.R) and sojo(https://github.com/zhenin/sojo/blob/master/R/sojo.R)packages.

def join_sumstat(SNP, D_neff_tmp, LD_matrix, var_y, betas):
  # Here I assume Dw * WW' * Dw explained in the paper is the LD matrix as suggested in the coco and sojo packages
  B = np.dot(np.sqrt(np.diag(D_neff_tmp).astype(float)), LD_matrix.astype(float), np.sqrt(np.diag(D_neff_tmp).astype(float)))
  #This is my version of the chol2inv indicated in the coco package
  #chol_matrix = np.linalg.cholesky(B)
  B_inv = np.linalg.inv(np.dot(B.T, B))
  new_betas = np.dot(np.dot(B_inv, np.diag(D_neff_tmp)), betas)
  new_se = np.diag(np.sqrt(np.abs(np.dot(B_inv,var_y))))
  new_z = (np.abs(new_betas/new_se))
  new_pval = scipy.stats.norm.sf(abs(new_z))*2
  d = {"SNP": SNP, "beta_cond_tmp": new_betas, "se_cond_tmp" : new_se, "pval_cond_tmp":new_pval }
  res_data = pd.DataFrame.from_dict(d)
  return res_data[res_data["SNP"]==SNP[0]]


### This is the second version of the joint algorithm. This follow the guideline reported here (https://www.mv.helsinki.fi/home/mjxpirin/GWAS_course/material/GWAS9.pdf)

def join_sumstat2(SNP, MAF, LD_matrix, betas, N, SE):
  MAF = MAF.astype(float)
  sc = np.sqrt(2*MAF*(1-MAF))
  b = betas*sc
  #This equation is reported also here (https://www.karger.com/Article/FullText/513303#ref4) as a way to compute the OLS.
  ls_sumstat = scipy.linalg.solve(LD_matrix, b) #computes scaled lambdas as R^-1 * b.s
  sigma2_J_sumstat = find_median(b**2 + N*sc**2*SE**2) - np.dot(b.T, ls_sumstat)
  ls_se_sumstat = np.sqrt(sigma2_J_sumstat/N * np.diag(scipy.linalg.inv(LD_matrix))) #SEs of scaled lambdas 
  ls_sumstat = (ls_sumstat/sc)
  ls_se_sumstat = (ls_se_sumstat/sc)
  new_z = (np.abs(ls_sumstat/ls_se_sumstat))
  new_pval = scipy.stats.norm.sf(abs(new_z))
  d = {"SNP": SNP, "beta_cond_tmp": ls_sumstat, "se_cond_tmp" : ls_se_sumstat, "pval_cond_tmp":new_pval }
  res_data = pd.DataFrame.from_dict(d)
  return res_data[res_data["SNP"]==SNP[0]]



# The only measures that needs to be calculated is the the 
# phenotype variance var_y if is not considered constant.
p_value_threshold = 5e-8
max_iter = 100
var_y = 1.3

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

# Calculate the variance weighted by N (D)
bim_uk_freq_filtered_SNP = (bim_uk_freq_filtered_SNP
                            .withColumn("var_af", (2*f.col("MAF")*(1-f.col("MAF"))))
                            .withColumn('D_neff', f.col("var_af") * f.col("n_total"))
)
bim_uk_freq_filtered_SNP = bim_uk_freq_filtered_SNP.withColumn('D_neff', 2*f.col("MAF")*(1-f.col("MAF")) * f.col("n_total"))
#bim_uk_freq_filtered_SNP = bim_uk_freq_filtered_SNP.withColumn('D_neff', ((var_y - (f.col("var_af") * f.col("beta")**2)))/(f.col("var_af") * f.col("se")**2) + 1)

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
                        .select("SNP" ,"pos","var_af","MAF","D_neff" ,"beta", "beta_cond", "se", "se_cond", "pval", "pval_cond", "n_total")
                        .toPandas())



###
# Create plink filtered file that will be used to calculate the LD matrix 
bim_uk_freq_filtered_SNP["SNP"].to_csv("test_data_single_window/variants_plink.csv", sep = " ", header = False)

input_bfile = "/Users/ba13/Desktop/Open_Target_Genetics/creation_pipeline/COJO_on_SPARK/test_data_single_window/ukb_v3_chr1.downsampled10k_window"
out_bfile= "/Users/ba13/Desktop/Open_Target_Genetics/creation_pipeline/COJO_on_SPARK/test_data_single_window/ukb_v3_chr1.downsampled10k_window_joint"
variants = "/Users/ba13/Desktop/Open_Target_Genetics/creation_pipeline/COJO_on_SPARK/test_data_single_window/variants_plink.csv"


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

SNP_name_LD = pd.read_csv("/Users/ba13/Desktop/Open_Target_Genetics/creation_pipeline/COJO_on_SPARK/test_data_single_window/ukb_v3_chr1.downsampled10k_window_joint.bim", sep = "\t", names = ["chr","SNP","rec","pos","A1","A2"])

bfile = "/Users/ba13/Desktop/Open_Target_Genetics/creation_pipeline/COJO_on_SPARK/test_data_single_window/ukb_v3_chr1.downsampled10k_window_joint"
out_ld = "/Users/ba13/Desktop/Open_Target_Genetics/creation_pipeline/COJO_on_SPARK/test_data_single_window/LD_SNP_test"

cmd_ld_create = [
        'plink',
        '--bfile', bfile,
        '--out', out_ld,
        '--r2 square'
    ]
cmd_str = ' '.join([str(x) for x in cmd_ld_create])

sp.call(cmd_str, shell=True)

ld_matrix = np.loadtxt('/Users/ba13/Desktop/Open_Target_Genetics/creation_pipeline/COJO_on_SPARK/test_data_single_window/LD_SNP_test.ld')
ld_matrix_names = pd.DataFrame(ld_matrix, index=SNP_name_LD["SNP"], columns=SNP_name_LD["SNP"])

variants_conditioned = pd.DataFrame({'SNP' : []})



#Either the minimum p-value is reached or the iteration reach a maximum
iters = 0
while(np.any(best_SNP_pvalue<p_value_threshold) and iters < max_iter):
    print("ok")
    
    for index, row in bim_uk_freq_filtered_SNP.iterrows():
      #Skip if the SNP to condition is included in the conditioned list
      if np.any(variants_conditioned["SNP"] == row["SNP"]):
        continue
        
      best_SNPs_cond_tmp = np.append(row["SNP"], best_SNPs_cond)
      select_rows = bim_uk_freq_filtered_SNP.loc[bim_uk_freq_filtered_SNP["SNP"].isin(best_SNPs_cond_tmp)]
      betas_slct = np.array(select_rows["beta"])
      D_neff_slct = np.array(select_rows["D_neff"])
      MAF_select = np.array(select_rows["MAF"])
      SE_select = np.array(select_rows["se"])
      N_slct = np.array(select_rows["n_total"])
      ld_matrix_slct_col = ld_matrix_names[best_SNPs_cond_tmp]
      ld_matrix_slct = ld_matrix_slct_col.loc[ld_matrix_slct_col.index.isin(best_SNPs_cond_tmp)].values
      out_col = ld_matrix_names[best_SNPs_cond_tmp]
      out = out_col.loc[out_col.index.isin(best_SNPs_cond_tmp)].values

      np.fill_diagonal(out,0) 
      
      # if the SNPs considered has a R2 higher than 0.9 with any of the top variants this is excluded from
      # further analysis
      if np.any(out>0.9):
          bim_uk_freq_filtered_SNP = bim_uk_freq_filtered_SNP.loc[bim_uk_freq_filtered_SNP['SNP'] != row["SNP"]]
          continue
        
      sum_stat_filtered_SNP_tmp = join_sumstat2(best_SNPs_cond_tmp, MAF_select, ld_matrix_slct,betas_slct, N_slct, SE_select)
      
      ### In case the first joint algorithm is used I need account for collinearity. 
      # ideally I need to use the VIF function although here I use a try/except that is quicker
      #try:
        #sum_stat_filtered_SNP_tmp = join_sumstat(SNP_slct, D_neff_slct, ld_matrix_slct, var_y, betas_slct)
      #except np.linalg.LinAlgError as err:
      #    print("collinearity problem")
      #    print(row["SNP"])
      #    bim_uk_freq_filtered_SNP_bestpval_wind = bim_uk_freq_filtered_SNP_bestpval_wind.loc[bim_uk_freq_filtered_SNP_bestpval_wind['SNP'] != row["SNP"]]
      #    continue
      
      bim_uk_freq_filtered_SNP.loc[bim_uk_freq_filtered_SNP["SNP"] == row["SNP"],"beta_cond"] = sum_stat_filtered_SNP_tmp["beta_cond_tmp"][0]
      bim_uk_freq_filtered_SNP.loc[bim_uk_freq_filtered_SNP["SNP"] == row["SNP"],"se_cond"] = sum_stat_filtered_SNP_tmp["se_cond_tmp"][0]
      bim_uk_freq_filtered_SNP.loc[bim_uk_freq_filtered_SNP["SNP"] == row["SNP"],"pval_cond"] = sum_stat_filtered_SNP_tmp["pval_cond_tmp"][0]

      #betas[best_SNPs_cond_tmp][0] = sum_stat_filtered_SNP_tmp["beta_cond_tmp"]
    
    print("end of cycle")
    
    #Here the selection of the best SNP is made again and the cycle is repeated
    best_SNP_ID, best_SNP_pos, best_SNP_pvalue, variants_conditioned =  select_best_SNP(bim_uk_freq_filtered_SNP, variants_conditioned)
    best_SNPs_cond = np.append(best_SNP_ID, best_SNPs_cond)
    iters = iters+1
    print(iters)
best_SNP_ID, best_SNP_pos, best_SNP_pvalue, variants_conditioned =  select_best_SNP(bim_uk_freq_filtered_SNP, variants_conditioned)

best_SNPs_cond = np.empty((0,1), str)
best_SNPs_cond = np.append(best_SNPs_cond, best_SNP_ID)

bim_uk_freq_filtered_SNP = bim_uk_freq_filtered_SNP.filter(f.col("pval")< 5e-8)
bim_uk_freq_filtered_SNP.repartition(1).write.format("csv").save("test_data/cojo_out/GCST002222_cojo_on_spark")
