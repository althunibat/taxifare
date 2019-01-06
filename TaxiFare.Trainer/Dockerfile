FROM microsoft/dotnet:2.2-runtime AS base
WORKDIR /app

FROM microsoft/dotnet:2.2-sdk AS build
WORKDIR /src
COPY TaxiFare.Trainer/Trainer.csproj TaxiFare.Trainer/
COPY TaxiFare.Model/TaxiFare.Model.csproj TaxiFare.Model/
RUN dotnet restore TaxiFare.Trainer/Trainer.csproj
COPY . .
WORKDIR /src/TaxiFare.Trainer
RUN dotnet build Trainer.csproj -c Release -o /app

FROM build AS publish
RUN dotnet publish Trainer.csproj -c Release -o /app

FROM base AS final
WORKDIR /app
COPY --from=publish /app .
ENTRYPOINT ["dotnet", "Trainer.dll"]
