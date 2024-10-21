use futures::{stream, FutureExt, StreamExt};
use models::{prompts, Llama321B};
use serde::{Deserialize, Serialize};
use std::io::{self, Write};
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

    print!(
        "Hello I'm your stock pilot, ask me anything about performance of stocks \n\
        Be sure to include the relevant date range\n\
        Your query: "
    );
    io::stdout().flush().unwrap();
    let mut query = String::new();
    io::stdin().read_line(&mut query).unwrap();
    let query = query.trim();
    let prompt = prompts::extract_params(query);
    println!("Query:\n {}", query);
    let result = llama_model.generate_with_default(&prompt, 100)?;
    println!("Result:\n {}", result);
    // Parse the JSON response
    let params: ExtractedParams = serde_json::from_str(&result.trim())?;
    println!("Extracted company names:\n {:?}", params);
    // Get the API token from the environment variable
    let api_token = std::env::var("STOCK_API_TOKEN")
        .expect("STOCK_API_TOKEN environment variable not set");
    let stock_api = api::StockApi::new(&api_token);

    let tickers = stream::iter(&params.company_names)
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
    let tickers_str = tickers.join(",");
    let data = stock_api
        .get_ticker_analytics(
            &tickers_str,
            &params.start_date,
            &params.end_date,
        )
        .await?;
    println!("Stock Data:\n {}", data);
    let query_prompt = prompts::analyze_query(&query, &data);
    let result = llama_model.generate_with_default(&query_prompt, 250)?;
    println!("Recommendation:\n {}", result);
    Ok(())
}
