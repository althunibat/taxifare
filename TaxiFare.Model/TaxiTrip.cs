namespace TaxiFare.Model
{
    public class TaxiTrip
    {
        public string VendorId { get; set; }
        public string RateCode { get; set; }
        public float PassengerCount { get; set; }
        public float TripTime { get; set; }
        public float TripDistance { get; set; }
        public string PaymentType { get; set; }
        public float FareAmount { get; set; }
    }
}
