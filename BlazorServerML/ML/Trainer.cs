using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers.LightGbm;

namespace BlazorServerML.ML
{
    public class Trainer
    {
        public static readonly string
            TRAIN_FILE = "housing.csv",
            TRAIN_PATH = Startup.PATH + "/Data/" + TRAIN_FILE,
            MODEL_FILE = "TrainedModel.zip",
            MODEL_PATH = Startup.PATH + "/Data/" + MODEL_FILE;

        readonly MLContext _ml = new(seed: 1);

        #region Prompt
        public delegate void PromptHandler(string message);
        public PromptHandler Prompt;
        void Echo(string message) => Prompt?.Invoke(message + "\n");
        #endregion

        // quality control
        double r2Average;
        const double ACCEPTED_ACCURACY = 0.7;

        public void CreateModel(string trainFile = null)
        {
            if (trainFile == null) {
                trainFile = TRAIN_PATH;
            }
            Echo($"Processing file: {Path.GetFileName(trainFile)} | {new FileInfo(trainFile).Length:#,###,###} Bytes");

            Echo("Loading data...");
            // 1. Load Data
            var trainingDataView = _ml.Data.LoadFromTextFile<HousingData>(
                path: trainFile,
                hasHeader: true,
                separatorChar: ',',
                allowQuoting: true,
                allowSparse: false);

            Echo("Building pipeline...");
            // 2. Build training pipeline
            var trainingPipeline = BuildTrainingPipeline();

            Echo("Training model...");
            // 3. Train Model
            var mlModel = TrainModel(trainingDataView, trainingPipeline);

            Echo("Evaluating model...");
            // 4. Evaluate quality of Model
            Evaluate(trainingDataView, trainingPipeline);

            Echo("Conclusion");
            // 5. Conclution
            Conclution(mlModel, trainingDataView.Schema);

            Echo("End of process");
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
            var model = trainingPipeline.Fit(trainingDataView);
            return model;
        }

        void Evaluate(IDataView trainingDataView, IEstimator<ITransformer> trainingPipeline)
        {
            // Cross-Validate with single dataset (since we don't have two datasets, one for training and for evaluate)
            // in order to evaluate and get the model's accuracy metrics

            Echo("\nCross-validating to get model's accuracy metrics");
            var crossValidationResults = _ml.Regression.CrossValidate(
                trainingDataView,
                trainingPipeline,
                numberOfFolds: 5,
                labelColumnName: "Label");
            PromptAverageMetrics(crossValidationResults);
        }

        void Conclution(ITransformer mlModel, DataViewSchema modelInputSchema)
        {
            if (r2Average < ACCEPTED_ACCURACY) {
                Echo($"\nThe trained model has low accuracy, less than {ACCEPTED_ACCURACY}, and will not be published.");
            }
            else {
                Echo("\nSaving the model...");
                _ml.Model.Save(mlModel, modelInputSchema, MODEL_PATH);
                Echo("\nReady Model");
            }
        }

        void PromptAverageMetrics(
            IEnumerable<TrainCatalogBase.CrossValidationResult<RegressionMetrics>> crossValidationResults
        )
        {
            var l1 = crossValidationResults.Select(r => r.Metrics.MeanAbsoluteError);
            var l2 = crossValidationResults.Select(r => r.Metrics.MeanSquaredError);
            var rms = crossValidationResults.Select(r => r.Metrics.RootMeanSquaredError);
            var lossFunction = crossValidationResults.Select(r => r.Metrics.LossFunction);
            var r2 = crossValidationResults.Select(r => r.Metrics.RSquared);
            // QC
            r2Average = r2.Average();

            Echo($"Metrics for Regression model");
            Echo($"Average L1 Loss:       {l1.Average():0.###}");
            Echo($"Average L2 Loss:       {l2.Average():0.###}");
            Echo($"Average RMS:           {rms.Average():0.###}");
            Echo($"Average Loss Function: {lossFunction.Average():0.###}");
            Echo($"Average R-squared:     {r2.Average():0.###}");
        }
    }
}
