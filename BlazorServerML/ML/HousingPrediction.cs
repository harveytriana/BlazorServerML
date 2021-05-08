using Microsoft.ML.Data;

namespace BlazorServerML.ML
{
    public class HousingPrediction
    {
        [ColumnName("Score")]
        public float PredictedPrice { get; set; }
    }
}
