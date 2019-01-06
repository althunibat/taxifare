using System.IO;
using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.Hosting;
using Microsoft.AspNetCore.Mvc;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.ML;
using Microsoft.ML.Core.Data;
using StackExchange.Redis;

namespace TaxiFare.Prediction
{
    public class Startup
    {
        public Startup(IConfiguration configuration)
        {
            Configuration = configuration;
        }

        public IConfiguration Configuration { get; }

        // This method gets called by the runtime. Use this method to add services to the container.
        public void ConfigureServices(IServiceCollection services)
        {
            var conn = ConnectionMultiplexer.Connect(Configuration["REDIS"]);
            var ml = new MLContext(0);
            
            services.AddSingleton<IConnectionMultiplexer>(c => conn);
            services.AddSingleton(c => ml);
            var prediction = ConfigurePrediction(conn, ml);
            services.AddSingleton(c => prediction);

            services.AddMvc()
                .SetCompatibilityVersion(CompatibilityVersion.Version_2_2);
            
        }

        private static ITransformer ConfigurePrediction(IConnectionMultiplexer conn, MLContext ml)
        {
            var db = conn.GetDatabase(1);
            ITransformer prediction;
            var bytes = (byte[]) db.StringGet("tm");
            using (var stream = new MemoryStream(bytes))
            {
                prediction = ml.Model.Load(stream);
            }

            return prediction;
        }

        // This method gets called by the runtime. Use this method to configure the HTTP request pipeline.
        public void Configure(IApplicationBuilder app, IHostingEnvironment env)
        {
            if (env.IsDevelopment())
            {
                app.UseDeveloperExceptionPage();
            }

            app.UseMvc();
        }
    }
}
