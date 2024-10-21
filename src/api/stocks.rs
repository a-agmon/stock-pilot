use serde::Deserialize;

#[derive(Clone)]
pub struct StockApi {
    client: reqwest::Client,
    token: String,
}
#[derive(Deserialize, Debug)]
struct SearchResult {
    #[serde(rename = "bestMatches")]
    best_matches: Vec<BestMatch>,
}
#[derive(Deserialize, Debug)]
struct BestMatch {
    #[serde(rename = "1. symbol")]
    symbol: String,
    #[serde(rename = "2. name")]
    name: String,
    #[serde(rename = "4. region")]
    region: String,
}
use std::collections::HashMap;

#[derive(Debug, Deserialize)]
struct Root {
    payload: Payload,
}

#[derive(Debug, Deserialize)]
struct Payload {
    RETURNS_CALCULATIONS: ReturnsCalculations,
}

#[derive(Debug, Deserialize)]
struct ReturnsCalculations {
    MEAN: HashMap<String, f64>,
    STDDEV: HashMap<String, f64>,
    CUMULATIVE_RETURN: HashMap<String, f64>,
}
impl StockApi {
    pub fn new(token: &str) -> Self {
        Self {
            client: reqwest::Client::new(),
            token: token.to_string(),
        }
    }
    pub async fn get_symbol(&self, company: &str) -> anyhow::Result<String> {
        let url = format!(
            "https://www.alphavantage.co/query?function=SYMBOL_SEARCH&keywords={}&apikey={}",
            company, self.token
        );
        let search_result =
            reqwest::get(url).await?.json::<SearchResult>().await?;
        let matches: Vec<BestMatch> = search_result
            .best_matches
            .into_iter()
            .filter(|c| c.region == "United States")
            .collect();
        if let Some(best_match) = matches.first() {
            Ok(best_match.symbol.clone())
        } else {
            anyhow::bail!("No matching symbol found for the company")
        }
    }

    pub async fn get_ticker_analytics(
        &self,
        tickers: &str,
        start_date: &str,
        end_date: &str,
    ) -> anyhow::Result<String> {
        let url = format!(
            "https://www.alphavantage.co/query?function=ANALYTICS_FIXED_WINDOW&SYMBOLS={}&RANGE={}&RANGE={}&INTERVAL=DAILY&OHLC=close&CALCULATIONS=MEAN,STDDEV,CUMULATIVE_RETURN&apikey={}",
            tickers, start_date, end_date, self.token
        );
        let response = reqwest::get(url).await?;
        let root: Root = response.json().await?;
        let mean = root.payload.RETURNS_CALCULATIONS.MEAN;
        let std = root.payload.RETURNS_CALCULATIONS.STDDEV;
        let cumulative_return =
            root.payload.RETURNS_CALCULATIONS.CUMULATIVE_RETURN;
        // Combine all maps to create a formatted string for each ticker
        let result = mean.iter()
            .map(|(ticker, mean_value)| {
                let std_value = std.get(ticker).unwrap_or(&0.0);
                let cumulative_return_value = cumulative_return.get(ticker).unwrap_or(&0.0);
                format!(
                    "{}: mean return: {:.4}%, std: {:.4}%, cumulative return: {:.4}%",
                    ticker,
                    mean_value * 100.0,
                    std_value * 100.0,
                    cumulative_return_value * 100.0
                )
            })
            .collect::<Vec<String>>()
            .join("\n");
        Ok(result.trim().to_string())
    }
}

// #[cfg(test)]
// mod tests {
//     use super::*;
//     use tokio;

//     #[tokio::test]
//     async fn test_get_symbol() {
//         // Create a mock API token (replace with a real token for integration testing)
//         let token = "-".to_string();
//         let stock_api = StockApi::new(&token);

//         // Test with a well-known company
//         let company = "Microsoft";
//         let result = stock_api.get_symbol(company).await;

//         assert!(result.is_ok(), "Failed to get symbol: {:?}", result.err());
//         let symbol = result.unwrap();
//         assert_eq!(symbol, "MSFT", "Unexpected symbol for Microsoft");

//         // Test with a non-existent company
//         let non_existent_company = "ThisCompanyDoesNotExist12345";
//         let result = stock_api.get_symbol(non_existent_company).await;

//         assert!(
//             result.is_err(),
//             "Expected an error for non-existent company"
//         );
//     }

//     #[tokio::test]
//     async fn test_get_ticker_analytics() {
//         let token = "_".to_string();
//         let stock_api = StockApi::new(&token);

//         let tickers = "DDOG,AAPL";
//         let result = stock_api.get_ticker_analytics(tickers).await;
//         assert!(
//             result.is_ok(),
//             "Failed to get ticker analytics: {:?}",
//             result.err()
//         );
//         println!("{:?}", result.unwrap());
//     }
// }
