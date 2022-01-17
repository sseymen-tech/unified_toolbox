# unified_toolbox
Code used in the paper "A unified optimization toolbox for solving popularity bias, fairness, and diversity in recommender systems" by Sinan Seymen, Himan Abdollahpouri and Edward C. Malthouse. 

Our main code can be found in comb_main.py. Only file required from the user is the utility file. Matrix V is basically |U|x|I| sized matrix with U being the user set and I being the item set. Matrix V represents all the values of user-item utilities, and can be obtained by any prediction algorithms. In our paper we have used SVD method. Gurobi academic license is used in this project.

This version is without any kind of approximations, and require a good memory to initialize. If memory is an issue, in the paper we have section discussing possible approximations that can be applied. Because this was intended as a short paper we did not include the approximation discussion. Clustering works well, and you can cluster items & users together to create smaller and easier to manage problems if yo uare having memory issues.


Link and citation: http://ceur-ws.org/Vol-2959/paper5.pdf

@inproceedings{seymen2021unified,
  title={A unified optimization toolbox for solving popularity bias, fairness, and diversity in recommender systems},
  author={Seymen, Sinan and Abdollahpouri, Himan and Malthouse, Edward Carl},
  booktitle={Workshop of Multi-Objective Recommender Systems (MORS’21), in conjunction with the 15th ACM Conference on Recommender Systems, RecSys},
  volume={21},
  pages={2021},
  year={2021}
}
