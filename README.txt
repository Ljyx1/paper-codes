The above code of D3QN series is the experimental code from the thesis ：
“ Multi-Agent Reinforcement Learning Resources Allocation 
Method Using Dueling Double Deep Q-Network in Vehicular Networks”


If you only need to test, just run the “Test_All“ program and the saved results will be stored in a txt file in the same folder.

Note that if you need to adjust the number of set transfer parameters, go to line 126 of the Environment_marl_test.py file and
find self.demand_size = int((4 * 190 + 300) * 8 * ?), than change the value at "?".

To ensure consistency with the structure of the original comparison paper, the code for the internal branch count of the test file has been retained.