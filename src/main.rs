use futures::{stream, FutureExt, StreamExt};
use models::{prompts, Llama321B};
use serde::{Deserialize, Serialize};

mod api;
mod models;

#[derive(Debug, Serialize, Deserialize)]
pub struct ExtractedParams {
    pub company_names: Vec<String>,
    pub start_date: String,
    pub end_date: String,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let mut llama_model = Llama321B::load().unwrap();
    let query = "Compare the performance of datadog and microsoft over last 7 days and recommend the better one";
    let prompt = prompts::extract_params(query);
    println!("Query:\n {}", query);
    let result = llama_model.generate_with_default(&prompt, 100)?;
    // Parse the JSON response
    let extracted_params: ExtractedParams = serde_json::from_str(&result)?;
    println!("Extracted company names:\n {:?}", extracted_params);
    let stock_api = api::StockApi::new("-");

    let tickers = stream::iter(&extracted_params.company_names)
        .then(|company_name| {
            let stock_api = stock_api.clone();
            async move { stock_api.get_symbol(company_name).await.ok() }
        })
        .collect::<Vec<_>>()
        .await
        .into_iter()
        .flatten()
        .collect::<Vec<_>>();

    println!("Found Tickers:\n {:?}", tickers);

    let data = stock_api.get_ticker_analytics(&tickers.join(",")).await?;
    println!("Stock Data:\n {}", data);
    let query_prompt = prompts::analyze_query(&query, &data);
    let mut llama_model = Llama321B::load().unwrap();
    let result = llama_model.generate_with_default(&query_prompt, 250)?;
    println!("Recommendation:\n {}", result);
    Ok(())
}
