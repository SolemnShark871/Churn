class models():
    def __init__(self):
        raise NotImplementedError

    def svm_churn(df, param_grid):

        ### Train test split FOR NUMERICAL ALGORITHMS: 20% test
        X = df.drop(['LABEL'],axis = 1)
        y = df['LABEL']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, stratify = y)

        svm = SVC()

        grid_search = GridSearchCV(svm, param_grid=param_grid,cv=5,verbose=2,scoring='accuracy', n_jobs=-1)
        grid_search.fit(X_train,y_train)

        best_model_params = grid_search.best_params_

        best_model =   SVC(kernel = best_model_params['kernel'], C = best_model_params['C'],
                        class_weight = best_model_params['class_weight'], gamma = best_model_params['gamma'])

        best_model.fit(X_train, y_train)

        y_pred = best_model.predict(X_test)

        print('----------Model report on all classes ----------')
        print(classification_report(y_test,y_pred, output_dict=True))

        return best_model, classification_report

    def svm_churn_tts(df, param_grid, test_train_split):

        ### Train test split FOR NUMERICAL ALGORITHMS: 20% test
        X = df.drop(['LABEL'],axis = 1)
        y = df['LABEL']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_train_split, stratify = y)

        svm = SVC()

        grid_search = GridSearchCV(svm, param_grid=param_grid,cv=5,verbose=2,scoring='accuracy', n_jobs=-1)
        grid_search.fit(X_train,y_train)

        best_model_params = grid_search.best_params_

        best_model =   SVC(kernel = best_model_params['kernel'], C = best_model_params['C'],
                        class_weight = best_model_params['class_weight'], gamma = best_model_params['gamma'])

        best_model.fit(X_train, y_train)

        y_pred = best_model.predict(X_test)

        print('----------Model report on all classes ----------')
        print(classification_report(y_test,y_pred, output_dict=False))

        return best_model, classification_report

    def log_reg_churn(df, grid_param):

        ### Train test split FOR NUMERICAL ALGORITHMS: 20% test
        X = df.drop(['LABEL'],axis = 1)
        y = df['LABEL']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify = y)

        log_reg = LogisticRegression()

        grid_search = GridSearchCV(estimator = log_reg, param_grid = grid_param, n_jobs = -1, cv = 5,
                                verbose = 2, return_train_score = True, scoring = "accuracy")
        grid_search.fit(X_train, y_train)

        best_model_params = grid_search.best_params_
        print(grid_search.best_params_)

        best_model = LogisticRegression(C = best_model_params['C'], penalty = best_model_params['penalty'], solver=best_model_params['solver'],
                                        max_iter=best_model_params['max_iter'], n_jobs = -1)


        best_model.fit(X_train, y_train)

        y_pred = best_model.predict(X_test)


        print('----------Model report on all classes ----------')
        print(classification_report(y_test,y_pred, output_dict=True))

        return best_model, classification_report

    def nn_churn(df, grid_param):
        X = df.drop(['LABEL'], axis = 1)
        y = df['LABEL']

        # Nested function to create model, required for KerasClassifier
        def create_model(
                        # Default values
                        activation: 'relu',
                        dropout_rate : 0,
                        init_mode: 'uniform',
                        #weight_constraint: 1,
                        optimizer: 'adam',
                        hiden_layers: 2,
                        units: [2, 2]) -> tf.keras.Sequential:
        
            # Create the model
            model = Sequential()
            model.add(Dense(X.shape[1], kernel_initializer =  init_mode, activation = activation))
        
            for i in range(hiden_layers):
                model.add(Dense(units = units[i], activation = activation))
            
            model.add(Dropout(dropout_rate))
            model.add(Dense(1, kernel_initializer = init_mode, activation = 'sigmoid'))
            model.compile(loss = "binary_crossentropy", optimizer = optimizer, metrics = ['accuracy'])
        
            return model

        #Model creation
        model_nn = KerasClassifier(build_fn = create_model)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, stratify = y)
        
        grid_search = GridSearchCV(estimator = model_nn, param_grid = grid_param, n_jobs = -1, 
                                cv = 5, verbose = 2, return_train_score = True, scoring = 'accuracy')
        
        grid_search.fit(X_train, y_train)
        
        best_model_params = grid_search.best_params_
        print(best_model_params)

        best_model = create_model(activation = best_model_params['activation'], dropout_rate = best_model_params['dropout_rate'],
                                init_mode = best_model_params['init_mode'], optimizer = best_model_params['optimizer'], 
                                hiden_layers = best_model_params['hiden_layers'], units = best_model_params['units'])
        

        best_model.fit(X_train, y_train, batch_size = best_model_params['batch_size'], epochs = best_model_params['epochs'], verbose = 2)

        y_pred = np.round(best_model.predict(X_test))


        print('----------Model report on all classes ----------')
        print(classification_report(y_test, y_pred, output_dict=True))

        return best_model, classification_report

## Creating the folder
dir_str = 'Results_' + str(datetime.datetime.now())
os.mkdir(dir_str)

## Preprocessing
preprocessed_data, common_empty_columns = preprocessing(df_retired, df_non_retired)
## Save
preprocessed_data.to_csv(os.path.join(dir_str, 'preprocessed_data.csv'))
pickle.dump(common_empty_columns, open(os.path.join(dir_str,'common_empty_columns.pkl'), 'wb'))


## Normalization
scaled_data, scaler = normalize_data(preprocessed_data)
## Save
scaled_data.to_csv(os.path.join(dir_str,'scaled_data.csv'))
pickle.dump(scaler, open(os.path.join(dir_str,'scaler.pkl'), 'wb'))


## PCA
pca_data, pca_ = pca(scaled_data, 250)
## Save
pca_data.to_csv(os.path.join(dir_str,'pca_data.csv'))
pickle.dump(pca_, open(os.path.join(dir_str,'pca_model.pkl'), 'wb'))

##---------------------------------------------------------------------------------------------------------
## ML SVM
grid_param_svm = {'C':  [0.0001, 0.001,0.01, 0.1, 1, 10, 100],
                  'gamma' : ['scale', 'auto', 0.0001, 0.001, 0.01, 0.1, 10, 100],
                  'class_weight': ['balanced', None],
                  'kernel' : ['sigmoid','poly','rbf']}
best_svm, report_svm = svm_churn(pca_data, grid_param_svm)
## Save
pickle.dump(best_svm, open(os.path.join(dir_str,'svm.sav'), 'wb'))
pickle.dump(report_svm, open(os.path.join(dir_str,'svm_metrics.pkl'), 'wb'))


##----------------------------------------------------------------------------------------------------------
## ML Logistic Regression
grid_param_lr = {"penalty": ["l1", "l2", "elasticnet", "none"],
              "C": [0.0001, 0.001, 0.01, 0.1, 1, 10, 100],
              "solver": ["newton-cg", "lbfgs", "liblinear", 'saga'],
              "max_iter": [500]
              }
best_lr, report_lr = log_reg_churn(pca_data, grid_param_lr)
## Save
pickle.dump(best_lr, open(os.path.join(dir_str,'lr.sav'), 'wb'))
pickle.dump(report_lr, open(os.path.join(dir_str,'lr_metrics.pkl'), 'wb'))

##----------------------------------------------------------------------------------------------------------
## ML Neural Network
grid_param_nn = {
    "activation": ['selu','softplus','softmax'],
    "init_mode": ['he_normal', 'glorot_normal'],
    "dropout_rate": [0.8],
    "units": [[8, 4]],
    "optimizer": ['RMSprop', 'Adam', 'SGD'],
    "hiden_layers": [2],
    "epochs": [15],
    "batch_size":  [128]
}
best_nn, report_nn = nn_churn(pca_data, grid_param_nn)
## Save
pickle.dump(best_nn, open(os.path.join(dir_str,'nn.sav'), 'wb'))
pickle.dump(report_nn, open(os.path.join(dir_str,'nn_metrics.pkl'), 'wb'))

#TTS for SVM

## Creating the folder
dir_str = 'Results_' + str(datetime.datetime.now())
os.mkdir(dir_str)

## Preprocessing
preprocessed_data, common_empty_columns = preprocessing(df_retired, df_non_retired)
## Save
preprocessed_data.to_csv(os.path.join(dir_str, 'preprocessed_data.csv'))
pickle.dump(common_empty_columns, open(os.path.join(dir_str,'common_empty_columns.pkl'), 'wb'))


## Normalization
scaled_data, scaler = normalize_data(preprocessed_data)
## Save
scaled_data.to_csv(os.path.join(dir_str,'scaled_data.csv'))
pickle.dump(scaler, open(os.path.join(dir_str,'scaler.pkl'), 'wb'))


## PCA
pca_data, pca_ = pca(scaled_data, 250)
## Save
pca_data.to_csv(os.path.join(dir_str,'pca_data.csv'))
pickle.dump(pca_, open(os.path.join(dir_str,'pca_model.pkl'), 'wb'))

##---------------------------------------------------------------------------------------------------------
## ML SVM
grid_param_svm = {'C':  [0.0001, 0.001,0.01, 0.1, 1, 10, 100],
                  'gamma' : ['scale', 'auto', 0.0001, 0.001, 0.01, 0.1, 10, 100],
                  'class_weight': ['balanced', None],
                  'kernel' : ['sigmoid','poly','rbf']}

## Perform 3 test-train splits

TTS = [0.2,0.1,0.05] #Try reversing this to see if that does anything 
for tts in TTS:

  #TODO try renaming these best_svmTTS (without for loop)
  print(tts*100)
  best_svm, report_svm = svm_churn_tts(pca_data, grid_param_svm, tts)
  #We need to clear the kernel for each training to avoid continuing 
  print()
  print()

  #TTS for SVM

## Creating the folder
dir_str = 'Results_' + str(datetime.datetime.now())
os.mkdir(dir_str)

## Preprocessing
preprocessed_data, common_empty_columns = preprocessing(df_retired, df_non_retired)
## Save
preprocessed_data.to_csv(os.path.join(dir_str, 'preprocessed_data.csv'))
pickle.dump(common_empty_columns, open(os.path.join(dir_str,'common_empty_columns.pkl'), 'wb'))


## Normalization
scaled_data, scaler = normalize_data(preprocessed_data)
## Save
scaled_data.to_csv(os.path.join(dir_str,'scaled_data.csv'))
pickle.dump(scaler, open(os.path.join(dir_str,'scaler.pkl'), 'wb'))


## PCA
pca_data, pca_ = pca(scaled_data, 250)
## Save
pca_data.to_csv(os.path.join(dir_str,'pca_data.csv'))
pickle.dump(pca_, open(os.path.join(dir_str,'pca_model.pkl'), 'wb'))

##---------------------------------------------------------------------------------------------------------
## ML SVM
grid_param_svm = {'C':  [0.0001, 0.001,0.01, 0.1, 1, 10, 100],
                  'gamma' : ['scale', 'auto', 0.0001, 0.001, 0.01, 0.1, 10, 100],
                  'class_weight': ['balanced', None],
                  'kernel' : ['sigmoid','poly','rbf']}

## Perform 3 test-train splits

TTS = [0.2,0.1,0.05] #Try reversing this to see if that does anything 


#TODO try renaming these best_svmTTS (without for loop)
print("20% Test")
best_svm20, report_svm20 = svm_churn_tts(pca_data, grid_param_svm, 0.2)
print()

print("10% Test")
best_svm10, report_svm10 = svm_churn_tts(pca_data, grid_param_svm, 0.1)
print()

print("5% Test")
best_svm5, report_svm5 = svm_churn_tts(pca_data, grid_param_svm, 0.05)

## Save
pickle.dump(best_svm, open(os.path.join(dir_str,'svm.sav'), 'wb'))
pickle.dump(report_svm, open(os.path.join(dir_str,'svm_metrics.pkl'), 'wb'))