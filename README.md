# Running machine learning from Blazor

*An example of model training and data prediction using ML.NET from a Blazor application.*

ML is one of those programming subjects whose skill requires not only knowledge in advanced programming, but also in mathematics, data analysis, and logical thinking. I believe that it is one of those paradigms that require a study that goes beyond what can traditionally be learned in networks and communities.

As a programmer and engineer I have often dealt with mathematics related to specific engineering problems, issues such as regressions, resolution of transcendental equations, algorithm programming, and the like. ML is specifically about finding a solution to a problem where there is no deductive linear solution, or it is not practical to treat it in those terms. On the contrary, ML applies inductive solutions, where the data goes first, and then the formulation. ML is not deterministic, in the sense that giving a specific solution, ML gives you a possible solution that must be the best.

I must confess that as a programmer I was interested in Python for this matter, and I learned that language. However, when I learned about ML.NET, I knew that I was going to go further with C#. The most popular is not equivalent to being the best. However, this opinion is personal, and perhaps biased in the sense that my experience in C# is extensive.

I clarify that the purpose of this publication is not to teach how to use ML.NET, but to show how to create a clean and effective code to run ML.NET in a Blazor app. In what way should we code the workflow concerned, and in turn as a way to write the predictions module in Blazor. The first thing we are going to establish is that ML should always be backend. Here's how I'll use a server-side Blazor app to tackle the matter.

So far, Blazor WebAssembly, even version 6, does not run ML.NET; which is somewhat natural. Although, we can through a Web API consume an ML.NET model from Blazor WASM.

> However, a WebAssembly client should be able to instantiate an ML.NET model and run it in its environment, an imperative in an application without an internet connection. I have sent a request to the .NET development team to enable a part of ML.NET that allows an ML model to be instantiated in Blazor WASM.

### The Blazor app

We are going to run ML.NET from a server blazor application. We need to add two references:

- Microsoft.ML
- Microsoft.ML.LightGbm

At the time of creating the solution I work with net5 and the versions of the references are 1.5.5.

> In particular, the second reference is due to the fact that it was suggested to me in my study; In general, most of the time it is not required. Perhaps illustrating that the number of trainers we can use in ML.NET is surprising.

The example deals with one of the classic examples of numerical regression in ML. Before, I would like to mention that if you venture into ML you will soon meet the [kaggle](https://www.kaggle.com) community, where you can download data sets for study, make comparisons, discuss issues. From there I took the data to build this example.

### The problem

In general, an ML problem starts from a data matrix whose columns can be of different data types, from which we want to predict values of one of the columns.

As I mentioned, I took a classic example of supervised numerical analysis. It consists of predicting the value of a house based on multiple features. However, the example is not a copy of some source, it was treated in logic, and very much in the modern C# style.

The data source is [California Housing Prices](https://www.kaggle.com/camnugent/california-housing-prices), from which in the data exploration I made a change in the labels of the `ocean_proximity` column , in particular I replaced the '>' character with LT (less than) so that it would not make noise when working with HTML as the trailing front end.

### Application architecture

As we know, ML consists of working on two scenarios, the Trainer, and the Predictor. The training workflow consists of:

- Data exploration
- Data upload
- Behavior channeling and algorithms
- Training
- Validation
- Publication

The detail of each of these steps is part of the general theory of ML. In particular, the third step, *Channeling behavior and algorithms*,  is what the data scientist does, and it is here that expertise and analytical skills make a difference, and make ML an art. The other points are certainly routine. We take into account that the training sequence can be repeated until the validation exceeds a qualitative threshold or, in other words, is acceptable.

> As a detail, in the pipeline I included the text column `OceanProximity`,  something that some examples about the problem presented here, which I have seen do not include. In transformations, there are several models to transform information from text to numbers.

To maintain order, organize the code for ML in a folder named ML.

### Data structures

In ML.NET data processing, two qualified data source structures are required. The first is used by training and the second by predictions. Regarding the example, we have:

**The HousingData class**

It corresponds to the map of the data matrix, together with the decorators required by ML.NET.

```csharp
using Microsoft.ML.Data;

namespace BlazorServerML.ML
{
    public class HousingData
    {
        [LoadColumn(0)]
        public float Longitude { get; set; }

        [LoadColumn(1)]
        public float Latitude { get; set; }

        [LoadColumn(2)]
        public float HousingMedianAge { get; set; }

        [LoadColumn(3)]
        public float TotalRooms { get; set; }

        [LoadColumn(4)]
        public float TotalBedrooms { get; set; }

        [LoadColumn(5)]
        public float Population { get; set; }

        [LoadColumn(6)]
        public float Households { get; set; }

        [LoadColumn(7)]
        public float MedianIncome { get; set; }

        // value to predict
        [LoadColumn(8), ColumnName("Label")]
        public float MedianHouseValue { get; set; }

        [LoadColumn(9)]
        public string OceanProximity { get; set; }
    }
}
```

**The HousingPrediction class**

It corresponds to the structure that will be used to generate a prediction, together with the decorators required by ML.NET.

```csharp
using Microsoft.ML.Data;

namespace BlazorServerML.ML
{
    public class HousingPrediction
    {
        [ColumnName("Score")]
        public float PredictedPrice { get; set; }
    }
}
```

### Training

It is the process of applying the ML workflow to a data set.

**The Trainer class**

```csharp
using System;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.ML;
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
        public delegate Task PromptHandler(string message);
        public PromptHandler Prompt;
        #endregion

        // quality control
        double r2Average;
        const double ACCEPTED_ACCURACY = 0.7;

        Predictor _predictor;

        public Trainer(Predictor predictor)
        {
            _predictor = predictor;
        }

        public async Task CreateModel(string trainFile = null)
        {
            try {
                if (trainFile == null) {
                    trainFile = TRAIN_PATH;
                }
                await Echo($"Processing file: {Path.GetFileName(trainFile)} |" +
                           $"| {new FileInfo(trainFile).Length:#,###,###} Bytes");

                await Echo("Loading data...");
                var trainingDataView = _ml.Data.LoadFromTextFile<HousingData>(
                    path: trainFile,
                    hasHeader: true,
                    separatorChar: ',',
                    allowQuoting: true,
                    allowSparse: false);

                await Echo("Building pipeline...");
                var trainingPipeline = BuildTrainingPipeline();

                await Echo("Training model...");
                var model = trainingPipeline.Fit(trainingDataView);

                await Echo("Evaluating model...");
                await Evaluate(trainingDataView, trainingPipeline);

                await Echo("Conclusion");
                if (r2Average < ACCEPTED_ACCURACY) {
                    await Echo($"The trained model has low accuracy, less than " +
                               $"{ACCEPTED_ACCURACY}, and will not be published.");
                }
                else {
                    await Echo($"The trained model has acceptable accuracy and will be published." +
                               $"Saving the model...");
                    _ml.Model.Save(model, trainingDataView.Schema, MODEL_PATH);
                    await Echo($"Model file was publishd as {MODEL_FILE} " +
                               $"| {new FileInfo(MODEL_PATH).Length:#,###,###} Bytes");

                    // update predictor engine
                    _predictor.LoadModel();
                    await Echo("Predictor engine was updated");
                }
                await Echo("End of process");
            }
            catch(Exception exception) {
                await Echo($"Exception:\n{exception.Message}");
                await Echo($"The process could not be completed.");
            }
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
            var ioColumns = new InputOutputColumnPair("OceanProximity", "OceanProximity");

            // data process configuration with pipeline data transformations
            var dataProcessPipeline = _ml
                .Transforms.Categorical.OneHotEncoding(new[] { ioColumns })
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

        async Task Evaluate(IDataView trainingDataView, IEstimator<ITransformer> trainingPipeline)
        {
            // Cross-Validate to evaluate and get the model's accuracy metrics
            await Echo("\nCross-validating to get model's accuracy metrics");
            var crossValidationResults = _ml.Regression.CrossValidate(
                trainingDataView,
                trainingPipeline,
                numberOfFolds: 5,
                labelColumnName: "Label");
            var l1 = crossValidationResults.Select(r => r.Metrics.MeanAbsoluteError);
            var l2 = crossValidationResults.Select(r => r.Metrics.MeanSquaredError);
            var ms = crossValidationResults.Select(r => r.Metrics.RootMeanSquaredError);
            var lf = crossValidationResults.Select(r => r.Metrics.LossFunction);
            var r2 = crossValidationResults.Select(r => r.Metrics.RSquared);
            // QC
            r2Average = r2.Average();
            // report
            await Echo($"Metrics for Regression model");
            await Echo($"Mean Absolute Error:     {l1.Average():0.###}");
            await Echo($"Mean Squared Error:      {l2.Average():0.###}");
            await Echo($"Root Mean Squared Error: {ms.Average():0.###}");
            await Echo($"Average Loss Function:   {lf.Average():0.###}");
            await Echo($"Average R-squared:       {r2.Average():0.###}\n");
        }

        async Task Echo(string message)
        {
            Prompt?.Invoke(message + "\n");
            // asynchronous for the reactivity of the interface that receives the event
            await Task.Delay(100);
        }
    }
}
```

**Details**

- In this code I add a quality control of the model, which determines that if the precision factor is less than 0.7, do not publish the model.

- The files lie in the same application, in the `Data` folder. The way the root of the path is determined is special, it comes from capturing the value in `Startup.cs`,  and it is preserved in the static variable, PATH.

- The evaluation uses *Cross-Validate*,  which in theory is more demanding and precise.

- I programmed a message event to report the flow that is being executed. It must be asynchronous because it runs from a web page (a Blazor component).

- The data file for the training can be changed, however, to simplify the example, this part was not added to the code. You can see that the `trainFile` parameter exists, which, if null, the data from *kaggle* will be used.

- `Predictor` is passed as a parameter because if the model changes we must update the prediction engine, since` Predictor` acts as a service.

### Prediction

It is the process of using a model generated by ML.

**The Predictor class**

```csharp
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
            if (_predictionEngine == null) {
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
```

**Details**

- The prediction engine exists as a service in the application, for which we register the service in Startup; `services.AddSingleton<Predictor>()`.

- Execute the creation of the prediction engine at the beginning or by explicit call from the trainer.

### The Blazor Components

Two Blazor pages are created, one for training and the other for getting predictions. We must bear in mind that the training component should be used by users with spatial privileges, since it is a specialized process.

#### The Trainer page

It presents the user interface so that through a command the training and publication process is executed.

```csharp
@page "/trainer"
@using BlazorServerML.ML

<h3>Trainer</h3>
<hr />
<h5 style="color:slategrey">Current data file: @Trainer.TRAIN_FILE</h5>

<button class="btn btn-primary"
        style="width:200px;margin-top:12px;"
        disabled="@disabled"
        @onclick="ExecuteTraining">
    Execute Training
</button>
<hr />
<br />
<h5 style="color:slategrey">Work flow</h5>
<div class="card bg-light p-3">
<pre>
@prompt
</pre>
</div>

@code {
    string prompt = "Ready";
    bool disabled;

    async Task ExecuteTraining()
    {
        disabled = true;
        prompt = "";

        var trainer = new Trainer();
        trainer.Prompt += AddPrompt;
        await trainer.CreateModel();

        disabled = false;
    }

    async Task AddPrompt(string text)
    {
        prompt += text;
        await InvokeAsync(StateHasChanged);
    }
}
```

*Page to run the training*

![](https://github.com/harveytriana/BlazorServerML/blob/master/Screens/bz_ml_1.png)

**Details**

- It subscribes to the `Prompt` event to present the messages sent in the process.

- As I mentioned above, I did not include any interface to load the data file. It is routine and is not the goal of this article.

#### The Predictor page

Basically it is a form in which a user specifies the parameters and executes the prediction.

```csharp
@page "/predictor"
@using BlazorServerML.ML
@inject Predictor _predictor

<h3>California Housing Prices</h3>
<hr />
<div class="row">
    <div class="col-12 col-md-6">
        <div class="form-group">
            <label>@Utils.SplitName(nameof(item.Longitude))</label>
            <input type="text" class="form-control" @bind="item.Longitude">
        </div>
    </div>
    <div class="col-12 col-md-6">
        <div class="form-group">
            <label>@Utils.SplitName(nameof(item.Longitude))</label>
            <input type="text" class="form-control" @bind="item.Latitude">
        </div>
    </div>
    <div class="col-12 col-md-6">
        <div class="form-group">
            <label>@Utils.SplitName(nameof(item.HousingMedianAge))</label>
            <input type="text" class="form-control" @bind="item.HousingMedianAge">
        </div>
    </div>
    <div class="col-12 col-md-6">
        <div class="form-group">
            <label>@Utils.SplitName(nameof(item.TotalRooms))</label>
            <input type="text" class="form-control" @bind="item.TotalRooms">
        </div>
    </div>
    <div class="col-12 col-md-6">
        <div class="form-group">
            <label>@Utils.SplitName(nameof(item.TotalBedrooms))</label>
            <input type="text" class="form-control" @bind="item.TotalBedrooms">
        </div>
    </div>
    <div class="col-12 col-md-6">
        <div class="form-group">
            <label>@Utils.SplitName(nameof(item.Population))</label>
            <input type="text" class="form-control" @bind="item.Population">
        </div>
    </div>
    <div class="col-12 col-md-6">
        <div class="form-group">
            <label>@Utils.SplitName(nameof(item.Households))</label>
            <input type="text" class="form-control" @bind="item.Households">
        </div>
    </div>
    <div class="col-12 col-md-6">
        <div class="form-group">
            <label>@Utils.SplitName(nameof(item.MedianIncome))</label>
            <input type="text" class="form-control" @bind="item.MedianIncome">
        </div>
    </div><div class="col-12 col-md-6">
        <div class="form-group">
            <label>@Utils.SplitName(nameof(item.MedianHouseValue))</label>
            <input type="text" class="form-control" @bind="predictionPrice" readonly="readonly">
        </div>
    </div>
    <div class="col-12 col-md-6">
        <div class="form-group">
            <label>@Utils.SplitName(nameof(item.OceanProximity))</label>
            <select class="form-control" @bind="@item.OceanProximity">
                <option value="NEAR BAY">NEAR BAY</option>
                <option value="NEAR OCEAN">NEAR OCEAN</option>
                <option value="INLAND">INLAND</option>
                <option value="LT 1H OCEAN">LT 1H OCEAN</option>
            </select>
        </div>
    </div>
    <div class="col-12">
        <button type="submit"
                class="btn btn-primary"
                style="width:200px;margin-top:12px;"
                @onclick="GetPrediction">
            Update Prediction
        </button>
    </div>
</div>

<hr />
<h4>Predicted Price: USD @predictionPrice.ToString("C")</h4>

@code {
    HousingData item;
    float predictionPrice;

    protected override void OnInitialized()
    {
        // setup with a real data
        item = new HousingData() {
            Longitude = -122.23F,
            Latitude = 37.88F,
            HousingMedianAge = 41F,
            TotalRooms = 880F,
            TotalBedrooms = 129F,
            Population = 322F,
            Households = 126F,
            MedianIncome = 8.3252F,
            OceanProximity = "NEAR BAY",
        };
        GetPrediction();
    }

    void GetPrediction()
    {
        var predictionData = _predictor.Predict(item);
        predictionPrice = predictionData != null ? predictionData.PredictedPrice : 0;
    }
}
```
*Page to run predictions*

![](https://github.com/harveytriana/BlazorServerML/blob/master/Screens/bz_ml_2.png)

**Details**

- The prediction engine is injected.

- The `OceanProximity` field has a special treatment. The column values were extracted and arranged in a select list. This favors the prediction since you are sending data that corresponds to the constants for this column. The values in the list are obtained in data exploration.

### Conclusions

The ML.NET library is a powerful tool for creating machine learning content for .NET applications. Using Blazor for server, we can create a modern application for machine learning, which runs both the training module and the prediction module.

---

This article is posted in »» https://www.blazorspread.net/blogview/running-machine-learning-from-blazor 

---

`MIT license. Author: Harvey Triana. Contact: admin @ blazorspread.net`
