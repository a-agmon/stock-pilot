use std::{io::Write, sync::Arc};

use futures::{stream, FutureExt, StreamExt};
use indicatif::{ProgressBar, ProgressStyle};
use models::{prompts, Llama321B};
use serde::{Deserialize, Serialize};
mod api;
mod models;
use console::{style, Term};
use tokio::sync::mpsc;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtractedParams {
    pub company_names: Vec<String>,
    pub start_date: String,
    pub end_date: String,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let mut llama_model = Llama321B::load()?;
    let term = Arc::new(Term::stdout());
    term.clear_screen()?;
    term.write_line(&format!("{}", style("Hello, I'm StockPilot, ask me anything about performance of stocks!").green().bright()))?;

    let (tx, mut rx) = mpsc::channel::<String>(100); // Spawn a task to handle output outside the loop
    let term_clone = term.clone();
    tokio::spawn(async move {
        while let Some(c) = rx.recv().await {
            term_clone.write_str(c.as_str()).unwrap();
        }
    });

    loop {
        term.write_line(&format!("{}", style("What's on your mind? (Be sure to include the relevant stock and date range):").bold().yellow()))?;
        let query = term.read_line()?;
        let pb = ProgressBar::new(2).with_style(ProgressStyle::with_template(
            "[{elapsed_precise}] {bar:25.cyan/blue} {pos:>7}/{len:7} {msg}",
        )?);
        pb.set_message("Extracting parameters...");
        let params = extract_params(&query, &mut llama_model).await?;
        pb.inc(1);
        let update_msg = format!(
            "Fetching tickers data for {} - {}...",
            &params.start_date, &params.end_date
        );
        pb.set_message(update_msg);
        let (tickers_str, ticker_analytics) =
            fetch_tickers_data(&params).await?;
        pb.inc(1);
        let finish_msg =
            format!("Analyzing tickers data for {}...", tickers_str);
        pb.finish_with_message(finish_msg);
        analyze_tickers_data(
            ticker_analytics,
            query.clone(),
            &mut llama_model,
            Some(tx.clone()),
        )
        .await?;
        //let elapsed = pb.elapsed();
        //pb.with_elapsed(elapsed).finish_with_message("Done!");
        // term.write_line(&format!(
        //     "\n Answer:\n {}",
        //     style(answer).cyan().bright()
        // ))?;
    }
}

async fn extract_params(
    query: &str,
    llama_model: &mut Llama321B,
) -> anyhow::Result<ExtractedParams> {
    let prompt = prompts::extract_params(query);
    let result = llama_model
        .generate_with_default(&prompt, 1.0, 100, None)
        .await?;
    let params: ExtractedParams = serde_json::from_str(&result.trim())?;
    Ok(params)
}

async fn analyze_tickers_data(
    tickers_data: String,
    query: String,
    llama_model: &mut Llama321B,
    stream_channel: Option<mpsc::Sender<String>>,
) -> anyhow::Result<String> {
    let query_prompt = prompts::analyze_query(&query, &tickers_data);
    llama_model
        .generate_with_default(&query_prompt, 1.5, 250, stream_channel)
        .await
}

async fn fetch_tickers_data(
    params: &ExtractedParams,
) -> anyhow::Result<(String, String)> {
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
    let ticker_analytics = stock_api
        .get_ticker_analytics(
            &tickers_str,
            &params.start_date,
            &params.end_date,
        )
        .await?;
    Ok((tickers_str, ticker_analytics))
}
