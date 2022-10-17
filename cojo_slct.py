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

##########
#This function is for dealing with multicollinearity in case the entire window is considered 
# for the joint analysis instead than for pairs of sites

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
  #df = df.join(f.broadcast(variants_conditioned), on = ["SNP"], how = "left_anti")
  R = df[:1]
  #R = df.head(1)[0]
  best_SNP_pos = R["pos"]
  best_SNP_ID = R["SNP"]
  best_SNP_pvalue = R["pval"]
  best_SNP_pos = int(best_SNP_pos)
  #best_SNP_conditioned = df.filter(f.col("pval") == best_SNP_pvalue).select("SNP")
  #variants_conditioned = variants_conditioned.union(best_SNP_conditioned)
  variants_conditioned = pd.concat([variants_conditioned,R])
  return [best_SNP_ID, best_SNP_pos, best_SNP_pvalue, variants_conditioned]

##########
# This function create a window of 4Mb (+/- 2Mb) around the selected variant
# Moreover it write the sets of variant that will later be used by plink to 
# calculate the LD matrix
def create_windows_plink(df, best_SNP_pos):
  df["pos"] = df["pos"].astype(int)

  df_wind = df.loc[(df["pos"] > best_SNP_pos - 2e6) &  (df["pos"] < best_SNP_pos + 2e6)]
  #df_wind = df.filter((f.col("pos") < best_SNP_pos + 2e6) & (f.col("pos") > best_SNP_pos - 2e6))
  df_wind_SNP = df_wind["SNP"]
  df_wind_SNP.to_csv("test_data/variants_plink.csv", sep = " ", header = False)
  #df_wind.select("SNP").coalesce(1).write.mode("overwrite").option("header", False).csv("test_data/variants.csv")
  return (df_wind)

#########
# This is the joint algorithm. It returns the lead variant adjusted for beta 
# se and p-value. What this function need to return is only the beta while the other
# measures can be calculated after the whole conditioning is made

def join_sumstat(SNP, D_neff_tmp, LD_matrix, var_y, betas):
  # Here I assume Dw * WW' * Dw explained in the paper
  # can be approximated by the LD correlation following what published here (insert link)
  B = np.dot(np.sqrt(np.diag(D_neff_tmp).astype(float)), LD_matrix.astype(float), np.sqrt(np.diag(D_neff_tmp).astype(float)))
  chol_matrix = np.linalg.cholesky(B)
  B_inv = np.linalg.inv(np.dot(B.T, B))
  new_betas = np.dot(np.dot(B_inv, np.diag(D_neff_tmp)), betas)
  new_se = np.diag(np.sqrt(np.abs(np.dot(B_inv,var_y))))
  new_z = (np.abs(new_betas/new_se))
  new_pval = scipy.stats.norm.sf(abs(new_z))*2
  d = {"SNP": SNP, "beta_cond_tmp": new_betas, "se_cond_tmp" : new_se, "pval_cond_tmp":new_pval }
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

bim_uk = spark.read.format("csv").load("test_data/ukb_v3_chr1.downsampled10k.reshaped.bim", sep = ",", schema = schema_bim)
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

# Consider only SNP that overlap between the summary stat and the reference
bim_uk_freq_filtered_SNP = bim_uk_freq.join(sumstat, on = ["SNP"], how = "inner")

#Calculate the variance (D)
bim_uk_freq_filtered_SNP = bim_uk_freq_filtered_SNP.withColumn('D_neff', 2*f.col("MAF")*(1-f.col("MAF")) * f.col("n_total"))

#Filter for SNPs that have no more than 0.2 of difference in the 
# allele frequencies between the reference anf the summary stat
bim_uk_freq_filtered_SNP = (bim_uk_freq_filtered_SNP
                        .withColumn("beta_cond", f.col("beta"))
                        .withColumn("pval_cond", f.col("pval"))
                        .withColumn("se_cond", f.col("se"))
                        .withColumn("diff_af", f.abs(f.col("MAF_sumstat")-f.col("MAF")))
                        .filter(f.col("diff_af")<0.2)
                        .select("SNP" ,"pos","MAF","D_neff" ,"beta", "beta_cond", "se", "se_cond", "pval", "pval_cond"))

bim_uk_freq_filtered_SNP = bim_uk_freq_filtered_SNP.sort(f.col("pval"))
iters = 1
last_p = 0

##### First cycle
#emptyRDD = spark.sparkContext.emptyRDD()
#schema_conditioned_SNP = t.StructType([
#  t.StructField('SNP', t.StringType(), True),
#  ])

#variants_conditioned = spark.createDataFrame(emptyRDD,schema_conditioned_SNP)
variants_conditioned = pd.DataFrame({'SNP' : []})

bim_uk_freq_filtered_SNP = bim_uk_freq_filtered_SNP.toPandas()

best_SNP_ID, best_SNP_pos, best_SNP_pvalue, variants_conditioned =  select_best_SNP(bim_uk_freq_filtered_SNP, variants_conditioned)
bim_uk_freq_filtered_SNP_bestpval_wind = create_windows_plink(bim_uk_freq_filtered_SNP, best_SNP_pos)
bfile = "/Users/ba13/Desktop/Open_Target_Genetics/creation_pipeline/COJO_on_SPARK/test_data/ukb_v3_chr1.downsampled10k"
out_ld = "/Users/ba13/Desktop/Open_Target_Genetics/creation_pipeline/COJO_on_SPARK/test_data/LD_SNP"
variants = "/Users/ba13/Desktop/Open_Target_Genetics/creation_pipeline/COJO_on_SPARK/test_data/variants_plink.csv"


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


bim_uk_freq_filtered_SNP_bestpval_wind_pd = bim_uk_freq_filtered_SNP_bestpval_wind
betas = np.array(bim_uk_freq_filtered_SNP_bestpval_wind_pd["beta_cond"])
D_neffs = np.array(bim_uk_freq_filtered_SNP_bestpval_wind_pd["D_neff"])

best_SNPs_cond = np.empty((0,1), int)

SNPs = np.array(bim_uk_freq_filtered_SNP_bestpval_wind_pd["SNP"])
top_SNP_index = np.where(SNPs == best_SNP_ID[0])[0][0]
best_SNPs_cond = np.append(best_SNPs_cond, top_SNP_index)

print("starting cycle")
max_iter = 10

#emptyRDD = spark.sparkContext.emptyRDD()
#schema_conditioned_SNP = t.StructType([
#  t.StructField('SNP', t.StringType(), True),
#  t.StructField('beta_cond_tmp', t.StringType(), True),
#  t.StructField('se_cond_tmp', t.StringType(), True),
#  t.StructField('pval_cond_tmp', t.StringType(), True),
  
#  ])

#snp_cycle_conditioned = spark.createDataFrame(emptyRDD,schema_conditioned_SNP)
#snp_cycle_conditioned = pd.DataFrame(columns=["SNP", "beta_cond_tmp", "se_cond_tmp", "pval_cond_tmp"])
while(best_SNP_pvalue.any() < p_value_threshold and iters < max_iter):
    t = 1
    #rows_LD_matrix = ld_filtered_SNP.count()
    #LD_matrix = np.ones((rows_LD_matrix, rows_LD_matrix))
    #np.fill_diagonal(LD_matrix, np.array(ld_filtered_SNP.select("R2").toPandas()))
    #print(LD_matrix)
    #betas = bim_uk_freq_filtered_SNP_bestpval_wind.select("beta_cond").toPandas()
    #D_neffs = np.array(bim_uk_freq_filtered_SNP_bestpval_wind_pd["D_neff"].values)
    for index, row in bim_uk_freq_filtered_SNP_bestpval_wind_pd.iterrows():
      print(index)
      if row["SNP"] != best_SNP_ID:
        best_SNPs_cond_tmp = np.append(best_SNPs_cond, index)
        betas_slct = (betas[best_SNPs_cond_tmp])
        D_neff_slct = D_neffs[best_SNPs_cond_tmp]
        ld_matrix_slct = ld_matrix[best_SNPs_cond_tmp, : ][: ,best_SNPs_cond_tmp]
        out = ld_matrix[best_SNPs_cond_tmp, : ][: ,best_SNPs_cond_tmp]
        np.fill_diagonal(out,0)
        #ld_matrix_slct = np.matrix([[1,ld_filtered_SNP_R2], [ld_filtered_SNP_R2, 1]]).astype(float)  
        SNP_slct = SNPs[best_SNPs_cond_tmp]
        if np.any(out>0.9):
          bim_uk_freq_filtered_SNP.loc[bim_uk_freq_filtered_SNP['SNP'] != row["SNP"]]
          #bim_uk_freq_filtered_SNP = bim_uk_freq_filtered_SNP.filter(f.col("SNP")!=row["SNP"])
          continue
        sum_stat_filtered_SNP_tmp = join_sumstat(SNP_slct, D_neff_slct, ld_matrix_slct, var_y, betas_slct)
        betas[best_SNPs_cond_tmp][0] = sum_stat_filtered_SNP_tmp["beta_cond_tmp"]
        
        #snp_cycle_conditioned = pd.concat([snp_cycle_conditioned, sum_stat_filtered_SNP_tmp])

        #sum_stat_filtered_SNP_tmp = (sum_stat_filtered_SNP_tmp
        #                            .withColumnRenamed("beta_cond", "beta_cond_tmp")
        #                            .withColumnRenamed("pval_cond", "pval_cond_tmp")
        #                            .withColumnRenamed("se_cond", "se_cond_tmp")
        #                            .filter(f.col("SNP") == row["SNP"])
        #                            .select("SNP", "beta_cond_tmp", "pval_cond_tmp", "se_cond_tmp")                           
        #)
        
        #bim_uk_freq_filtered_SNP = bim_uk_freq_filtered_SNP.join(sum_stat_filtered_SNP_tmp, on = ["SNP"], how = "outer")
        #bim_uk_freq_filtered_SNP = (bim_uk_freq_filtered_SNP
        #                            .withColumn("beta_cond_merge", f.when(f.col("beta_cond_tmp").isNull(),f.col("beta_cond")).otherwise(f.col("beta_cond_tmp")))
        #                            .withColumn("pval_cond_merge", f.when(f.col("pval_cond_tmp").isNull(),f.col("pval_cond")).otherwise(f.col("pval_cond_tmp")))
        #                            .withColumn("se_cond_merge", f.when(f.col("se_cond_tmp").isNull(),f.col("se_cond")).otherwise(f.col("se_cond_tmp")))
        #                            .drop(sum_stat_filtered_SNP_tmp.beta_cond_tmp)
        #                            .drop(sum_stat_filtered_SNP_tmp.pval_cond_tmp)
        #                            .drop(sum_stat_filtered_SNP_tmp.se_cond_tmp)
        #)
    
    print("end of cycle")
    bim_uk_freq_filtered_SNP = bim_uk_freq_filtered_SNP.iloc[bim_uk_freq_filtered_SNP["SNP"] == best_SNP_ID]["beta_cond"] = sum_stat_filtered_SNP_tmp["beta_cond_tmp"][0]
    bim_uk_freq_filtered_SNP = bim_uk_freq_filtered_SNP.iloc[bim_uk_freq_filtered_SNP["SNP"] == best_SNP_ID]["se_cond"] = sum_stat_filtered_SNP_tmp["se_cond_tmp"][0]
    bim_uk_freq_filtered_SNP = bim_uk_freq_filtered_SNP.iloc[bim_uk_freq_filtered_SNP["SNP"] == best_SNP_ID]["pval_cond"] = sum_stat_filtered_SNP_tmp["pval_cond_tmp"][0]

    #bim_uk_freq_filtered_SNP = (bim_uk_freq_filtered_SNP
    #                            .withColumn("beta_cond", f.when(f.col("SNP") == best_SNP_ID, sum_stat_filtered_SNP_tmp["beta_cond_tmp"][0]))
    #                            .withColumn("pval_cond", f.when(f.col("SNP") == best_SNP_ID, sum_stat_filtered_SNP_tmp["pval_cond_tmp"][0]))
    #                            .withColumn("se_cond", f.when(f.col("SNP") == best_SNP_ID, sum_stat_filtered_SNP_tmp["se_cond_tmp"][0]))
    #                      )
    #bim_uk_freq_filtered_SNP = bim_uk_freq_filtered_SNP.join(snp_cycle_conditioned, on = ["SNP"], how = "outer")
    #bim_uk_freq_filtered_SNP = (bim_uk_freq_filtered_SNP
    #                               .withColumn("beta_cond_merge", f.when(f.col("beta_cond_tmp").isNull(),f.col("beta_cond")).otherwise(f.col("beta_cond_tmp")))
    #                                .withColumn("pval_cond_merge", f.when(f.col("pval_cond_tmp").isNull(),f.col("pval_cond")).otherwise(f.col("pval_cond_tmp")))
    #                                .withColumn("se_cond_merge", f.when(f.col("se_cond_tmp").isNull(),f.col("se_cond")).otherwise(f.col("se_cond_tmp")))
    #                                .drop(sum_stat_filtered_SNP_tmp.beta_cond_tmp)
    #                                .drop(sum_stat_filtered_SNP_tmp.pval_cond_tmp)
    #                                .drop(sum_stat_filtered_SNP_tmp.se_cond_tmp)
    #                             )
    best_SNP_ID, best_SNP_pos, best_SNP_pvalue, variants_conditioned =  select_best_SNP(bim_uk_freq_filtered_SNP, variants_conditioned)
    bim_uk_freq_filtered_SNP_bestpval_wind = create_windows_plink(bim_uk_freq_filtered_SNP, best_SNP_pos)
    cmd_str = ' '.join([str(x) for x in cmd_bim_extract])
    sp.call(cmd_str, shell=True)
    ld_matrix = np.loadtxt('/Users/ba13/Desktop/Open_Target_Genetics/creation_pipeline/COJO_on_SPARK/test_data/LD_SNP.ld')
    #bim_uk_freq_filtered_SNP_bestpval_wind_pd = bim_uk_freq_filtered_SNP_bestpval_wind.toPandas()
    #betas = da.array(bim_uk_freq_filtered_SNP_bestpval_wind.select("beta_cond").collect()).reshape(-1)
    #D_neffs = da.array(bim_uk_freq_filtered_SNP_bestpval_wind.select("D_neff").collect()).reshape(-1)
    #SNPs = da.array(bim_uk_freq_filtered_SNP_bestpval_wind.select("SNP").collect()).reshape(-1)
    betas = np.array(bim_uk_freq_filtered_SNP_bestpval_wind_pd["beta_cond"])
    D_neffs = np.array(bim_uk_freq_filtered_SNP_bestpval_wind_pd["D_neff"])
    SNPs = np.array(bim_uk_freq_filtered_SNP_bestpval_wind_pd["SNP"])
    top_SNP_index = np.where(SNPs == best_SNP_ID[0])[0][0]
    best_SNPs_cond = np.append(top_SNP_index, best_SNPs_cond)
    iters = iters+1
    print(iters)


bim_uk_freq_filtered_SNP = bim_uk_freq_filtered_SNP.filter(f.col("pval")< 5e-8)
bim_uk_freq_filtered_SNP.repartition(1).write.format("csv").save("test_data/cojo_out/GCST002222_cojo_on_spark")
