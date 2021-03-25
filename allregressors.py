class regression():
    def __init__(self,X_train, X_test,y_train,y_test, n_estimators,random_state,kind,degree=1,kernel='rbf'):
        self.n_estimators=n_estimators
        self.random_state=random_state
        self.X_train=X_train
        self.X_test=X_test
        self.y_train=y_train
        self.y_test=y_test
        self.kind=kind
        self.degree=degree
        self.kernel=kernel

    def predict(self):
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.linear_model import LinearRegression
        from sklearn.tree import DecisionTreeRegressor
        from sklearn.preprocessing import PolynomialFeatures
        from sklearn.svm import SVR
        if self.kind=='RF':
            regressor=RandomForestRegressor(n_estimators = self.n_estimators, random_state = self.random_state)
            print("random forest is called")
        elif self.kind=='LR':
            regressor= LinearRegression()
            print("Linear Regression is called")
        elif self.kind=='DT':
            regressor = DecisionTreeRegressor(random_state = self.random_state)
        elif self.kind=='PR':
            poly_reg = PolynomialFeatures(degree = self.degree)
            self.X_train = poly_reg.fit_transform(self.X_train)
            regressor = LinearRegression()
            self.X_test=poly_reg.fit_transform(self.X_test)
        elif self.kind=='SVR':
            pass
        #     regressor = SVR(self.kernel)
        regressor.fit(self.X_train,self.y_train)
        predict=regressor.predict(self.X_test)
        return predict