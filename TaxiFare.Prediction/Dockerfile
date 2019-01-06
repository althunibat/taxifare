FROM microsoft/dotnet:2.2-aspnetcore-runtime AS base
WORKDIR /app
EXPOSE 80

FROM microsoft/dotnet:2.2-sdk AS build
WORKDIR /src
COPY TaxiFare.Prediction/TaxiFare.Prediction.csproj TaxiFare.Prediction/
RUN dotnet restore TaxiFare.Prediction/TaxiFare.Prediction.csproj
COPY . .
WORKDIR /src/TaxiFare.Prediction
RUN dotnet build TaxiFare.Prediction.csproj -c Release -o /app

FROM build AS publish
RUN dotnet publish TaxiFare.Prediction.csproj -c Release -o /app

FROM base AS final
WORKDIR /app
COPY --from=publish /app .
ENTRYPOINT ["dotnet", "TaxiFare.Prediction.dll"]
