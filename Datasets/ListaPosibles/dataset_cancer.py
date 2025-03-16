from ucimlrepo import fetch_ucirepo 
  
# fetch dataset 
breast_cancer_wisconsin_diagnostic = fetch_ucirepo(id=17) 
  
# data (as pandas dataframes) 
X = breast_cancer_wisconsin_diagnostic.data.features 
y = breast_cancer_wisconsin_diagnostic.data.targets 

  
# variable information 
# print(list(breast_cancer_wisconsin_diagnostic.variables.name))

X.to_csv("Cancer_features.csv", index = False)
y.to_csv("Cancer_target.csv", index = False)








