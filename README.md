# TickerPilot

TickerPilot is an educational repo that demonstrates how to build an intelligent stock market analysis tool powered by Llama 3.21B LLM and based on  [Alpha Vantagen Stock API](https://www.alphavantage.co/).  
It allows users to ask questions about stock performance and receive insightful answers based on real-time market data.  
Its implemented in Rust using [Hugging Face Candle lib](https://github.com/huggingface/candle) and the model was fine-tuned using [Unsloth](https://unsloth.ai/).


![animated gif](assets/tickerpilot.gif)


## Features

- Natural language interface for querying stock market information
- Automatic extraction of relevant parameters (company names, date ranges) from user queries
- Real-time fetching of stock data using an API
- AI-powered analysis of stock performance and trends
- User-friendly command-line interface with progress indicators

## How it works

1. The user inputs a question about stock performance.
2. TickerPilot extracts relevant parameters from the query using the Llama 3 21B model.
3. It fetches real-time stock data for the specified companies and date range.
4. The AI model analyzes the data and generates a comprehensive answer to the user's query.

## Getting Started

1. Ensure you have Rust installed on your system.
2. Clone this repository.
3. Set the `STOCK_API_TOKEN` environment variable with your Stock API token.
4. Run `cargo build` to compile the project.
5. Execute `cargo run` to start TickerPilot.

## Dependencies

- Rust
- tokio (for async runtime)
- serde (for serialization/deserialization)
- indicatif (for progress bars)
- console (for terminal styling)
- anyhow (for error handling)
- Llama 3 21B model (for natural language processing)

## License

[Add your chosen license here]

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
