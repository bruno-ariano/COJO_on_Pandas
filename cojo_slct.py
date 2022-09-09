from pyspark.sql import SparkSession
from pyspark.sql.types import *
import pyspark.sql.functions as F
from pyspark.sql.window import Window
from pyspark import SparkConf
import pandas as pd
import numpy
import scipy.special as sps


global spark
sparkConf = (
        SparkConf()
        .set("spark.hadoop.fs.gs.requester.pays.mode", "AUTO")
        .set("spark.hadoop.fs.gs.requester.pays.project.id", cfg.project.id)
        .set("spark.sql.broadcastTimeout", "36000")
    )

    # establish spark connection
spark = SparkSession.builder.config(conf=sparkConf).master("yarn").getOrCreate()

print('Spark version: ', spark.version)

genotype_ref = 'file:///home/ba13/fine-mapping/genetics-finemapping/LD_reference/ukb_v3_chr9.downsampled10k'
window_gwas = 
def sumstat_ukBio(genotype_matrix, target_SNPs):
    ukbio_geno = spark.read.format('plink').load(genotype_matrix)
    gwas_sumstat = (spark.read.format('parquet')
                    .load(target_SNPs)
                    )
    ukbio_gwas_data = ukbio_geno.join(gwas_sumstat, on = ['snp'])
    return ukbio_gwas_data


def normalize_beta(ukbio_gwas_data):




#parquet_file = ...
def cojo_slct(window, red_ld, p_value):

stepwise.fwd <- function(y, X, p.thresh = 1e-4, plot.path = F, ring.i = NULL){
  #Does stepwise forward selection where at most one variant is included in S at each iteration
  ## INPUT
  # y, quantitative trait for n individuals
  # X, genotypes for n individuals x p SNPs
  # p.thresh, P-value threshold that is used for deciding whether to include variant in S
  # plot.path, TRUE / FALSE, whether produces a plot of every step of the algorithm
  # ring.i, set of indexes of SNPs that should always be highlighted in plots by ringing
  ## OUTPUT
  # chosen.i, indexes of chosen variants in S, 
  # chosen.p, the P-values at the iteration when each chosen variant was chosen, 
  # next.p, the smallest P-value left outside S when finished.
  
  p = ncol(X) #number of SNPs
  pval = apply(X, 2, function(x){summary(lm(y ~ x))$coeff[2,4]}) #start from marginal P-values
  chosen.i = c() #collect here the chosen SNPs
  chosen.p = c() #collect here the P-values from the iteration where the choice was made
  ii = 0 #iteration index
  while(min(pval) < p.thresh){ #continue as long as something is below the threshold
    ii = ii + 1 
    chosen.i[ii] = which.min(pval)[1] # add 1 SNP with min P-val to the chosen ones...
    chosen.p[ii] = pval[chosen.i[ii]] #... and store its P-value at this iteration.
    #test other SNPs except the already chosen ones -- and include the chosen ones as covariates
    tmp = apply(X[,-chosen.i], 2, function(x){summary(lm(y ~ x + X[,chosen.i]))$coeff[2,4]}) 
    pval[-chosen.i] = tmp # we have P-values for other SNPs except already chosen ones in 'tmp'
    pval[chosen.i] = 1 #mark chose ones by P-value = 1 for the while-loop condition
  }
  return(list(chosen.i = chosen.i, chosen.p = chosen.p, next.p = min(pval)))
}

