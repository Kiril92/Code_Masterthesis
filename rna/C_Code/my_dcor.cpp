//[[Rcpp::plugins(cpp11)]]

#include <iostream> 
#include <fstream>
#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]

using namespace Rcpp;
using namespace arma;
using namespace std;

//[[Rcpp::export]]
NumericMatrix euclidean_dist(NumericVector x){
  
  const int len=x.size();
  NumericMatrix result(len, len);
  
  double x_i;
  double a;
  int i,j;
  
  for(i=0;i<len;++i){
    x_i = x[i];
    for(j=0; j<len; ++j){
      a=fabs(x[j]-x_i);
      result(i,j)=a;
      result(j,i)=a;
    }
  }
  
  return result;
}

//[[Rcpp::export]]
NumericMatrix my_dcor(NumericMatrix X, string logfile) {
  
  ofstream myfile;
  
  const int ncl=X.ncol(), nrw=X.nrow();
  int i,j,k;
  NumericMatrix result(ncl,ncl);
  NumericVector a_rowmean(nrw), b_rowmean(nrw);
  NumericVector m_dvarX(ncl);
  NumericMatrix m_cov(ncl,ncl);
  double dcov,dvarX,dvarY;
  
  // Berechnung aller A Matrizen
  List A_matrices(ncl);
  
  
  // Schritt 1: Berechnung der Varianzen je Spalte
  for(i=0; i<ncl; ++i){
    
    // Fortschrittsanzeige
    if(i%10==0){
      myfile.open(logfile);
      myfile << "Berechnung A-Matritzen: " << i <<"\n";
      myfile.close();
    }
    
    NumericMatrix a = euclidean_dist(X(_,i));
    
    // Umwandeln von Rcpp::NumericMatric in Arma::mat
    mat aa(a.begin(),nrw,nrw,false);
    
    // Spaltensummen (= Zeilensummen.t())
    rowvec ma = mean(aa,0);
    mat A = aa.each_row() - ma;   
    A = A.each_col() - ma.t();
    A = A + mean(ma);
    
    A_matrices[i] = A;
    m_dvarX[i] = sqrt( mean(mean(square(A))) );
  }
  
  
  // Schritt 2: Berechnung der paarweisen Covarianzen
  for(i=0; i<ncl; ++i){
      
    if(i%10==0){
      myfile.open(logfile);
      myfile << "Cov - Iteration: " << i <<"\n";
      myfile.close();
    }
      
    mat A = A_matrices[i];
    for(j=i; j<ncl; ++j){
      mat B = A_matrices[j];
      m_cov(i,j) = sqrt( mean(mean(A % B)));
    }
  }
  
  for(i=0; i<ncl; ++i){
    
    if(i%10==0){
      myfile.open(logfile);
      myfile << "Dcor - Iteration: " << i <<"\n";
      myfile.close();
    }
    // mat A = A_matrices[i];
    
    for(j=i; j<ncl; ++j){
      // mat B = A_matrices[j];
      // dvarX = m_dvarX[i];
      // dvarY = m_dvarX[j];
      // dcov = sqrt( mean(mean(A % B)) );
      result(j,i) = m_cov(i,j)/sqrt(m_dvarX[i] * m_dvarX[j]);
      result(i,j) = result(j,i);
    }
  }
  
  
  return result;
}
