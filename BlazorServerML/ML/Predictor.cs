// ===========================
// BlazorSpread.net
// ===========================
using System;
using Microsoft.ML;

namespace BlazorServerML.ML
{
    public class Predictor
    {
        PredictionEngine<HousingData, HousingPrediction> _predictionEngine;

        public Predictor()
        {
            LoadModel();
        }

        public HousingPrediction Predict(HousingData input)
        {
            if (_predictionEngine == null) {// unexpected
                return null;
            }
            return _predictionEngine.Predict(input);
        }

        public void LoadModel()
        {
            _predictionEngine = null;
            try {
                var mlContext = new MLContext();
                var mlModel = mlContext.Model.Load(Trainer.MODEL_PATH, out _);
                _predictionEngine = mlContext.Model.CreatePredictionEngine<HousingData, HousingPrediction>(mlModel);

            }
            catch (Exception exception) {
                Console.WriteLine("LoadModel Exception:\n" + exception.Message);
            }
        }
    }
}
