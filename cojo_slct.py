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


#def calc_VIF(x):
#  vif= pd.DataFrame()
#  vif["VIF"]=[variance_inflation_factor(x.values,i) for i in range(x.shape[1])]
#  vif['variables']=x.columns

spark = SparkSession.builder.config("spark.some.config.option", "some-value").master("local").getOrCreate()


def find_median(values_list):
  median = np.median(values_list)
  return round(float(median),2)

med_find=f.udf(find_median,t.FloatType())

def select_best_SNP(df, variants_conditioned):
  df = df.join(variants_conditioned, on = ["SNP"], how = "left_anti")
  best_SNP_pvalue = df.agg({"pval": 'min'}).collect()[0][0]
  best_SNP_pos = df.filter(f.col("pval") == best_SNP_pvalue).select("pos").collect()[0][0]
  best_SNP_pos = int(best_SNP_pos)
  best_SNP_ID = df.filter(f.col("pval") == best_SNP_pvalue).select("SNP").collect()[0][0]
  best_SNP_conditioned = df.filter(f.col("pval") == best_SNP_pvalue).select("SNP")
  variants_conditioned = variants_conditioned.union(best_SNP_conditioned)
  return [best_SNP_ID, best_SNP_pos, best_SNP_pvalue, variants_conditioned]
  
  
def create_windows_plink(df, best_SNP_pos):
  df_wind = df.filter((f.col("pos") < best_SNP_pos + 1e6) & (f.col("pos") > best_SNP_pos - 1e6))
  df_wind.select("SNP").coalesce(1).write.mode("overwrite").option("header", False).csv("test_data/variants.csv")
  return (df_wind)


def join_sumstat(SNP, D_neff_tmp, LD_matrix, var_y, betas):
    ####Here I assume Dw * WW' * Dw can be interpreted as the LD correlation between SNPs as highlighted in the github
  B = np.dot(np.sqrt(np.diag(D_neff_tmp).astype(float)), LD_matrix.astype(float), np.sqrt(np.diag(D_neff_tmp).astype(float)))
  chol_matrix = np.linalg.cholesky(B)
  B_inv = np.linalg.inv(np.dot(B.T, B))
  new_betas = np.dot(np.dot(B_inv, np.diag(D_neff_tmp)), betas)
  new_se = np.diag(np.sqrt(np.abs(np.dot(B_inv,var_y))))
  new_z = (np.abs(new_betas/new_se))
  new_pval = scipy.stats.norm.sf(abs(new_z))*2
  #new_pval = (np.abs(new_betas/new_se))
  d = {"SNP": SNP,"beta": betas, "beta_cond": new_betas, "se_cond" : new_se, "pval_cond":new_pval }
  res_data = pd.DataFrame.from_dict(d)
  res_data = spark.createDataFrame(res_data)
  return res_data


#establish spark connection
p_value_threshold = 1e-5
max_iter = 100
var_y = 1.3


#loading files
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


bim_uk = spark.read.format("csv").load("test_data/ukb_v3_chr1.downsampled10k.reshaped.bim", sep = ",", schema = schema_bim)
freq_uk = spark.read.format("csv").load("test_data/ukb_v3_chr1.downsampled10k.reshaped.frq", sep = ",", schema = schema_freq)
freq_uk = freq_uk.select("SNP", "MAF")
bim_uk_freq = bim_uk.join(freq_uk, on = ["SNP"])


sumstat = spark.read.format("parquet").load("test_data/GCST002222.parquet")
sumstat = (sumstat
          .withColumn("SNP", f.concat_ws(":", f.col("chrom"), f.col("pos"), f.col("ref"), f.col("alt")))
          .select("SNP","beta","se","pval","n_total")
)

bim_uk_freq_filtered_SNP = bim_uk_freq.join(sumstat, on = ["SNP"], how = "inner")
bim_uk_freq_filtered_SNP = bim_uk_freq_filtered_SNP.withColumn('D_neff', 2*f.col("MAF")*(1-f.col("MAF")) * f.col("n_total"))
bim_uk_freq_filtered_SNP = (bim_uk_freq_filtered_SNP
                         .withColumn("beta_cond", f.col("beta"))
                         .withColumn("pval_cond", f.col("pval"))
                         .withColumn("se_cond", f.col("se"))
                         .select("SNP" ,"pos","MAF","D_neff" ,"beta", "beta_cond", "se", "se_cond", "pval", "pval_cond"))
  
iters = 1
last_p = 0

##### First cycle
emptyRDD = spark.sparkContext.emptyRDD()
schema_conditioned_SNP = t.StructType([
  t.StructField('SNP', t.StringType(), True),
  ])
variants_conditioned = spark.createDataFrame(emptyRDD,schema_conditioned_SNP)

best_SNP_ID, best_SNP_pos, best_SNP_pvalue, variants_conditioned =  select_best_SNP(bim_uk_freq_filtered_SNP, variants_conditioned)
bim_uk_freq_filtered_SNP_bestpval_wind = create_windows_plink(bim_uk_freq_filtered_SNP, best_SNP_pos)
bim_uk_freq_filtered_SNP_bestpval_wind = bim_uk_freq_filtered_SNP.filter((f.col("pos") < best_SNP_pos + 1e6) & (f.col("pos") > best_SNP_pos - 1e6))
bfile = "/Users/ba13/Desktop/Open_Target_Genetics/creation_pipeline/COJO_on_SPARK/test_data/ukb_v3_chr1.downsampled10k"
out_ld = "/Users/ba13/Desktop/Open_Target_Genetics/creation_pipeline/COJO_on_SPARK/test_data/LD_SNP"
variants = "/Users/ba13/Desktop/Open_Target_Genetics/creation_pipeline/COJO_on_SPARK/test_data/variants.csv/*.csv"
cmd_bim_extract = [
        'plink',
        '--bfile', bfile,
        '--extract',  variants,
        '--r2 square',
        '--allow-extra-chr',
        '--out', out_ld
    ]
cmd_str = ' '.join([str(x) for x in cmd_bim_extract])

sp.call(cmd_str, shell=True)

ld_matrix = np.loadtxt('/Users/ba13/Desktop/Open_Target_Genetics/creation_pipeline/COJO_on_SPARK/test_data/LD_SNP.ld')

#to_exclude = np.unique(np.where((ld_matrix >0.9) & (ld_matrix != 1))[0])

#print(to_exclude.shape)
#print(ld_matrix.shape)
#ld_matrix_nozero = np.delete(np.delete(ld_matrix,to_exclude, 1),to_exclude, 0)
#ld_matrix_nozero = ld_matrix

#snp_matrix = pd.read_csv('/Users/ba13/Desktop/Open_Target_Genetics/creation_pipeline/COJO_on_SPARK/test_data/LD_SNP.snplist', header= None)
#snp_matrix_nozero = spark.createDataFrame(snp_matrix.iloc[snp_matrix.index.difference(to_exclude)])

#snp_matrix_nozero = snp_matrix_nozero.withColumnRenamed("0", "SNP")
#bim_uk_freq_filtered_SNP_bestpval_wind = bim_uk_freq_filtered_SNP_bestpval_wind.join(snp_matrix_nozero, on = ["SNP"])

#bim_uk_freq_filtered_SNP_bestpval_wind.show()
  

bim_uk_freq_filtered_SNP_bestpval_wind_pd = bim_uk_freq_filtered_SNP_bestpval_wind.toPandas()
betas = np.array(bim_uk_freq_filtered_SNP_bestpval_wind_pd["beta_cond"])
D_neffs = np.array(bim_uk_freq_filtered_SNP_bestpval_wind_pd["D_neff"])

best_SNPs_cond = np.empty((0,1), int)

SNPs = np.array(bim_uk_freq_filtered_SNP_bestpval_wind_pd["SNP"])
top_SNP_index = np.where(SNPs == best_SNP_ID)[0][0]
best_SNPs_cond = np.append(best_SNPs_cond, top_SNP_index)

print("starting cycle")
max_iter = 10


while(best_SNP_pvalue < p_value_threshold and iters < max_iter):
    t = 1
    #rows_LD_matrix = ld_filtered_SNP.count()
    #LD_matrix = np.ones((rows_LD_matrix, rows_LD_matrix))
    #np.fill_diagonal(LD_matrix, np.array(ld_filtered_SNP.select("R2").toPandas()))
    #print(LD_matrix)
    #betas = bim_uk_freq_filtered_SNP_bestpval_wind.select("beta_cond").toPandas()
    #D_neffs = np.array(bim_uk_freq_filtered_SNP_bestpval_wind_pd["D_neff"].values)
    for index, row in bim_uk_freq_filtered_SNP_bestpval_wind_pd.iterrows():
      if row["SNP"] != best_SNP_ID:
        best_SNPs_cond_tmp = np.append(best_SNPs_cond, index)
        betas_slct = (betas[best_SNPs_cond_tmp])
        D_neff_slct = D_neffs[best_SNPs_cond_tmp]
        ld_matrix_slct = ld_matrix[best_SNPs_cond_tmp, : ][: ,best_SNPs_cond_tmp]
        if np.any(ld_matrix_slct>0.9):
          continue
        #ld_matrix_slct = np.matrix([[1,ld_filtered_SNP_R2], [ld_filtered_SNP_R2, 1]]).astype(float)  
        SNP_slct = SNPs[best_SNPs_cond_tmp]
        sum_stat_filtered_SNP_tmp = join_sumstat(SNP_slct, D_neff_slct, ld_matrix_slct, var_y, betas_slct)
        bim_uk_freq_filtered_SNP = bim_uk_freq_filtered_SNP.join(sum_stat_filtered_SNP_tmp, on = ["SNP"], how = "left")
    #res_data = res_data.union(joint_results)
    
    print("end of cycle")
    best_SNP_ID, best_SNP_pos, best_SNP_pvalue, variants_conditioned =  select_best_SNP(bim_uk_freq_filtered_SNP, variants_conditioned)
    print(best_SNP_ID)
    bim_uk_freq_filtered_SNP_bestpval_wind = create_windows_plink(bim_uk_freq_filtered_SNP, best_SNP_pos)
    bim_uk_freq_filtered_SNP_bestpval_wind = bim_uk_freq_filtered_SNP.filter((f.col("pos") < best_SNP_pos + 1e6) & (f.col("pos") > best_SNP_pos - 1e6))
    #bim_uk_freq_filtered_SNP_bestpval_wind = bim_uk_freq_filtered_SNP_bestpval_wind.union(bim_uk_freq_filtered_SNP.filter(f.col("row_index") == best_SNPs_cond))
    cmd_str = ' '.join([str(x) for x in cmd_bim_extract])
    sp.call(cmd_str, shell=True)
    ld_matrix = np.loadtxt('/Users/ba13/Desktop/Open_Target_Genetics/creation_pipeline/COJO_on_SPARK/test_data/LD_SNP.ld')

    bim_uk_freq_filtered_SNP_bestpval_wind_pd = bim_uk_freq_filtered_SNP_bestpval_wind.toPandas()
    betas = np.array(bim_uk_freq_filtered_SNP_bestpval_wind_pd["beta_cond"])
    D_neffs = np.array(bim_uk_freq_filtered_SNP_bestpval_wind_pd["D_neff"])
    SNPs = np.array(bim_uk_freq_filtered_SNP_bestpval_wind_pd["SNP"])
    
    top_SNP_index = np.where(SNPs == best_SNP_ID)[0][0]
    best_SNPs_cond = np.append(best_SNPs_cond, top_SNP_index)
    iters = iters+1
    print(iters)
bim_uk_freq_filtered_SNP = bim_uk_freq_filtered_SNP.filter(f.col("pval")< 5e-5)
bim_uk_freq_filtered_SNP.repartition(1).write.format("csv").save("test_data/cojo_out/GCST002222_cojo_on_spark")

