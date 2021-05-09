using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers.LightGbm;

namespace BlazorServerML.ML
{
    public class Trainer
    {
        public static readonly string
            TRAIN_DATA = Startup.PATH + "/Data/housing.csv",
            MODEL_PATH = Startup.PATH + "/ML/TrainedModel.zip";

        readonly MLContext _ml = new(seed: 1);

        // messages to container
        public delegate void PromptHandler(string message);
        public PromptHandler Prompt;
        //  Prompt?.Invoke("READY | KEYS Q-A: Player1, KEYS O-L: Player2");

        public void CreateModel()
        {
            // 1. Load Data
            var trainingDataView = _ml.Data.LoadFromTextFile<HousingData>(
                path: TRAIN_DATA,
                hasHeader: true,
                separatorChar: ',',
                allowQuoting: true,
                allowSparse: false);

            // 2. Build training pipeline
            var trainingPipeline = BuildTrainingPipeline();

            // 3. Train Model
            var mlModel = TrainModel(trainingDataView, trainingPipeline);

            // 4. Evaluate quality of Model
            Evaluate(trainingDataView, trainingPipeline);

            // 5. Save model
            SaveModel(mlModel, trainingDataView.Schema);
        }

        IEstimator<ITransformer> BuildTrainingPipeline()
        {
            var feactures = new string[] {
                "OceanProximity",
                "Longitude",
                "Latitude",
                "HousingMedianAge",
                "TotalRooms",
                "TotalBedrooms",
                "Population",
                "Households",
                "MedianIncome"
            };
            var inputOutputColumns = new InputOutputColumnPair("OceanProximity", "OceanProximity");

            // data process configuration with pipeline data transformations 
            var dataProcessPipeline = _ml
                .Transforms.Categorical.OneHotEncoding(new[] { inputOutputColumns })
                .Append(_ml.Transforms.Concatenate("Features", feactures));

            // set the training algorithm 
            var trainer = _ml.Regression.Trainers.LightGbm(new LightGbmRegressionTrainer.Options() {
                NumberOfIterations = 200,
                LabelColumnName = "Label",
                FeatureColumnName = "Features"
            });

            var trainingPipeline = dataProcessPipeline.Append(trainer);

            return trainingPipeline;
        }

        ITransformer TrainModel(IDataView trainingDataView, IEstimator<ITransformer> trainingPipeline)
        {
            Console.WriteLine("\nTraining model");

            var model = trainingPipeline.Fit(trainingDataView);

            Console.WriteLine("\nEnd of training process");
            return model;
        }

        void Evaluate(IDataView trainingDataView, IEstimator<ITransformer> trainingPipeline)
        {
            // Cross-Validate with single dataset (since we don't have two datasets, one for training and for evaluate)
            // in order to evaluate and get the model's accuracy metrics
            Console.WriteLine("\nCross-validating to get model's accuracy metrics");
            var crossValidationResults = _ml.Regression.CrossValidate(trainingDataView, trainingPipeline, numberOfFolds: 5, labelColumnName: "Label");
            PrintRegressionFoldsAverageMetrics(crossValidationResults);
        }

        void SaveModel(ITransformer mlModel, DataViewSchema modelInputSchema)
        {
            Console.WriteLine("\nSave the model");
            _ml.Model.Save(mlModel, modelInputSchema, MODEL_PATH);
        }

        void PrintRegressionMetrics(RegressionMetrics metrics)
        {
            Console.WriteLine($"*************************************************");
            Console.WriteLine($"*       Metrics for Regression model      ");
            Console.WriteLine($"*------------------------------------------------");
            Console.WriteLine($"*       LossFn:        {metrics.LossFunction:0.##}");
            Console.WriteLine($"*       R2 Score:      {metrics.RSquared:0.##}");
            Console.WriteLine($"*       Absolute loss: {metrics.MeanAbsoluteError:#.##}");
            Console.WriteLine($"*       Squared loss:  {metrics.MeanSquaredError:#.##}");
            Console.WriteLine($"*       RMS loss:      {metrics.RootMeanSquaredError:#.##}");
            Console.WriteLine($"*************************************************");
        }

        void PrintRegressionFoldsAverageMetrics(IEnumerable<TrainCatalogBase.CrossValidationResult<RegressionMetrics>> crossValidationResults)
        {
            var L1 = crossValidationResults.Select(r => r.Metrics.MeanAbsoluteError);
            var L2 = crossValidationResults.Select(r => r.Metrics.MeanSquaredError);
            var RMS = crossValidationResults.Select(r => r.Metrics.RootMeanSquaredError);
            var lossFunction = crossValidationResults.Select(r => r.Metrics.LossFunction);
            var R2 = crossValidationResults.Select(r => r.Metrics.RSquared);

            Console.WriteLine($"*************************************************************************************************************");
            Console.WriteLine($"*       Metrics for Regression model      ");
            Console.WriteLine($"*------------------------------------------------------------------------------------------------------------");
            Console.WriteLine($"*       Average L1 Loss:       {L1.Average():0.###} ");
            Console.WriteLine($"*       Average L2 Loss:       {L2.Average():0.###}  ");
            Console.WriteLine($"*       Average RMS:           {RMS.Average():0.###}  ");
            Console.WriteLine($"*       Average Loss Function: {lossFunction.Average():0.###}  ");
            Console.WriteLine($"*       Average R-squared:     {R2.Average():0.###}  ");
            Console.WriteLine($"*************************************************************************************************************");
        }
    }
}
