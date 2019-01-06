FROM microsoft/dotnet:2.2-runtime AS base
WORKDIR /app

FROM microsoft/dotnet:2.2-sdk AS build
WORKDIR /src
COPY TaxiFare.Trainer/TaxiFare.Trainer.csproj TaxiFare.Trainer/
RUN dotnet restore TaxiFare.Trainer/TaxiFare.Trainer.csproj
COPY . .
WORKDIR /src/TaxiFare.Trainer
RUN dotnet build TaxiFare.Trainer.csproj -c Release -o /app

FROM build AS publish
RUN dotnet publish TaxiFare.Trainer.csproj -c Release -o /app

FROM base AS final
WORKDIR /app
COPY --from=publish /app .
ENTRYPOINT ["dotnet", "TaxiFare.Trainer.dll"]
