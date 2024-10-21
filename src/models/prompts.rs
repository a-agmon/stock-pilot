pub fn extract_params(prompt: &str) -> String {
    format!(
        r#"
        <|begin_of_text|><|start_header_id|>system<|end_header_id|>
        You are an expert at extracting parameters from text. 
        You are given a query that mentions company names and dates. 
        Extract the parameters: company_names, start_date, and end_date from the user's query. 
        Do not include any additional text in your response.
      <|eot_id|><|start_header_id|>user<|end_header_id|>
      Today is {today}. Extract the parameters in this query to json according to the following format:  {{\"company_names\": [\"AAPL\", \"GOOGL\"], \"start_date\": \"2024-01-01\", \"end_date\": \"2024-01-31\"}}.
      Query:{prompt}
      <|eot_id|><|start_header_id|>assistant<|end_header_id|>"#,
        today = chrono::Local::now().format("%Y-%m-%d").to_string(),
        prompt = prompt
    )
}

pub fn analyze_query(query: &str, data: &str) -> String {
    format!(
        r#"
          <|begin_of_text|><|start_header_id|>system<|end_header_id|>
          You are a stock market expert. 
          You get information in a json format about companies, their mean and cumulative returns for a given period and uses this information to answer a query.
        <|eot_id|><|start_header_id|>user<|end_header_id|> 
        Given the following data, please answer the following query:{query}
        {data}
        <|eot_id|><|start_header_id|>assistant<|end_header_id|> "#,
        query = query,
        data = data
    )
}
