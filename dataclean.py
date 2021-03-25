class dataclean():
    def __init__(self,dataset,x_columns,y_columns,test_size,random_state):
        self.x_columns=x_columns
        self.y_columns=y_columns
        self.test_size=test_size
        self.dataset=dataset
        self.random_state=random_state
        self.X=self.dataset[self.x_columns]
        self.y=self.dataset[self.y_columns]

    def traintest(self):
        import numpy as np
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size = self.test_size, random_state = self.random_state)
        return {'X_train':np.array(X_train).reshape(-1,1),'X_test':np.array(X_test).reshape(-1,1),'y_train':np.array(y_train).reshape(-1,1),'y_test':np.array(y_test).reshape(-1,1)}