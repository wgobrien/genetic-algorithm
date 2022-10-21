from ga import GeneticAlgorithm
from math_functions import *

if __name__=='__main__':
    # path to models
    models_path = os.path.join(os.path.dirname(__file__), '../models/')

    # Load svr model
    #svr1_path = os.path.join(models_path, 'svr_model.pkl')
    #SVR = joblib.load(svr1_path)

    # Load scaler models for predictions
    X_scale_path = os.path.join(models_path, 'scalers/X_scale.pkl')
    y_scale_path = os.path.join(models_path, 'scalers/y_scale.pkl')
    #X_scale = joblib.load(X_scale_path)
    #y_scale = joblib.load(y_scale_path)

    # Create GA object
    parameters = ['LaserPowerHatch', 'LaserSpeedHatch', 'HatchSpacing', 'LaserPowerContour']
    boundaries = [(100, 400), (600, 1200), (.1,.25), (30,200)]
    #ga = GeneticAlgorithm(SVR, parameters, boundaries, X_scale, y_scale, pop_size=50)

    # Test make prediction
    # predict = ga.model_predict({'LaserPowerHatch':300, 'LaserSpeedHatch':1200, 'HatchSpacing': .15, 'LaserPowerContour': 140})

    # Run the algorithm to find optimal parameter set
    # best_performer, converge_hist = ga.run(mode='minimize'
    #                                        , select='rank'
    #                                        , mutation_rate='dynamic'
    #                                        , generations=1000
    #                                        , exploration=.3
    #                                        , keep_top=1
    #                                        , verbose=True)
    # ga.export(best_performer) # best is optional, can have export run the algorithm instead
    ga = GeneticAlgorithm(rastrigin, ['x','y','z'], [(-5.12,5.12),(-5.12,5.12),(-5.12,5.12)], pop_size=100)
    b, n = ga.run(mode='minimize'
                  , select='rank'
                  , mutation_rate='dynamic'
                  , generations=2000
                  , exploration=.35
                  , keep_top=1
                  , verbose=True)
    ga.export(b)
