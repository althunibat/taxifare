FROM microsoft/dotnet:2.2-aspnetcore-runtime AS base
WORKDIR /app
EXPOSE 80

FROM microsoft/dotnet:2.2-sdk AS build
WORKDIR /src
COPY TaxiFare.Prediction/Prediction.csproj TaxiFare.Prediction/
COPY TaxiFare.Model/TaxiFare.Model.csproj TaxiFare.Model/
RUN dotnet restore TaxiFare.Prediction/Prediction.csproj
COPY . .
WORKDIR /src/TaxiFare.Prediction
RUN dotnet build Prediction.csproj -c Release -o /app

FROM build AS publish
RUN dotnet publish Prediction.csproj -c Release -o /app

FROM base AS final
WORKDIR /app
COPY --from=publish /app .
ENTRYPOINT ["dotnet", "Prediction.dll"]
