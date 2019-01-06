using System.Collections.Generic;
using System.IO;
using System.Threading.Tasks;
using Microsoft.AspNetCore.Mvc;
using Microsoft.ML;
using Microsoft.ML.Core.Data;
using Microsoft.ML.Runtime.Api;
using Microsoft.ML.Runtime.Data;
using StackExchange.Redis;
using TaxiFare.Model;

namespace TaxiFare.Prediction.Controllers
{
    [Route("api/[controller]")]
    [ApiController]
    public class PredictionsController : ControllerBase
    {
        private readonly MLContext _mlContext;
        private readonly IConnectionMultiplexer _redis;
        private readonly ITransformer _prediction;
        public PredictionsController(MLContext mlContext, IConnectionMultiplexer redis, ITransformer prediction)
        {
            _mlContext = mlContext;
            _redis = redis;
            _prediction = prediction;
        }

        [HttpPost]
        public IActionResult Post([FromBody] TaxiTrip trip)
        {
            // Create prediction engine related to the loaded trained model
            var function = _prediction.MakePredictionFunction<TaxiTrip, TaxiTripFarePrediction>(_mlContext);
            //Score
            var farePrediction = function.Predict(trip);
            return Ok(farePrediction.FareAmount);
        }

        public class TaxiTripFarePrediction
        {
            [ColumnName("Score")]
            public float FareAmount { get; set; }
        }
    }
}
