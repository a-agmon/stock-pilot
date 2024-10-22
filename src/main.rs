use futures::{stream, FutureExt, StreamExt};
use indicatif::{ProgressBar, ProgressStyle};
use models::{prompts, Llama321B};
use serde::{Deserialize, Serialize};
mod api;
mod models;
use console::{style, Term};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtractedParams {
    pub company_names: Vec<String>,
    pub start_date: String,
    pub end_date: String,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let mut llama_model = Llama321B::load()?;
    let term = Term::stdout();
    term.clear_screen()?;
    term.write_line(&format!("{}", style("Hello, I'm TickerPilot, ask me anything about performance of stocks!").green().bright()))?;

    loop {
        term.write_line(&format!("{}", style("What would you like to know about the stock market? (Be sure to include the relevant stock and date range):").bold().yellow()))?;
        let query = term.read_line()?;
        let pb = ProgressBar::new(3).with_style(ProgressStyle::with_template(
            "[{elapsed_precise}] {bar:40.cyan/blue} {pos:>7}/{len:7} {msg}",
        )?);
        pb.set_message("Extracting parameters...");
        let params = extract_params(&query, &mut llama_model)?;
        pb.inc(1);
        let update_msg = format!(
            "Fetching tickers data for {} - {}...",
            &params.start_date, &params.end_date
        );
        pb.set_message(update_msg);
        let tickers_data = fetch_tickers_data(&params).await?;
        pb.inc(1);
        pb.set_message("Analyzing tickers data...");
        let answer =
            analyze_tickers_data(tickers_data, query.clone(), &mut llama_model)
                .await?;
        pb.inc(1);
        let elapsed = pb.elapsed();
        pb.with_elapsed(elapsed).finish_with_message("Done!");
        term.write_line(&format!(
            "\n Answer:\n {}",
            style(answer).cyan().bright()
        ))?;
    }
}

fn extract_params(
    query: &str,
    llama_model: &mut Llama321B,
) -> anyhow::Result<ExtractedParams> {
    let prompt = prompts::extract_params(query);
    let result = llama_model.generate_with_default(&prompt, 100)?;
    let params: ExtractedParams = serde_json::from_str(&result.trim())?;
    Ok(params)
}

async fn analyze_tickers_data(
    tickers_data: String,
    query: String,
    llama_model: &mut Llama321B,
) -> anyhow::Result<String> {
    let query_prompt = prompts::analyze_query(&query, &tickers_data);
    llama_model.generate_with_default(&query_prompt, 250)
}

async fn fetch_tickers_data(
    params: &ExtractedParams,
) -> anyhow::Result<String> {
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

    let tickers_str = tickers.join(",");
    stock_api
        .get_ticker_analytics(
            &tickers_str,
            &params.start_date,
            &params.end_date,
        )
        .await
}
