import json
import argparse
import numpy as np
import copy

def generate_timestamps(data, session_lambda, request_lambda, session_scale=1.0, request_scale=1.0):
    """
    Generate timestamps for ShareGPT conversations

    Args:
        data: The parsed JSON data
        session_lambda: Lambda parameter for inter-session arrival times (average rate of sessions)
        request_lambda: Lambda parameter for inter-request arrival times (average rate of requests)
            LAMBDA: number of events in a fixed interval
        session_scale: Scaling factor for inter-session times
        request_scale: Scaling factor for inter-request times
            Scaling factor to adjust poisson generated intervals - adjust the overall time between requests
            without changing distribution
    """
    timestamped_data = copy.deepcopy(data)
    current_time = 0
    
    for session_idx, session in enumerate(timestamped_data):
        # sample random from session poisson
        if session_idx > 0:
            session_interval = np.random.poisson(session_lambda) * session_scale
            current_time += session_interval
        
        session_start_time = current_time
        session["session_start_time"] = session_start_time
        
        # Process messages within the conversation
        message_time = session_start_time
        
        for msg_idx, msg in enumerate(session.get("conversations", [])):
            # Only generate new timestamp for human messages
            if msg.get("from") == "human":
                if msg_idx > 0:
                    # sample random from request poisson
                    msg_interval = np.random.poisson(request_lambda) * request_scale
                    message_time += msg_interval
                
                msg["timestamp"] = message_time
            # GPT messages get the same timestamp as the previous human message
            elif msg.get("from") == "gpt":
                msg["timestamp"] = message_time
    
    return timestamped_data

def main():
    parser = argparse.ArgumentParser(description="Add timestamps to ShareGPT conversations")
    parser.add_argument("input_file", help="Input JSON file")
    parser.add_argument("output_file", help="Output JSON file")
    parser.add_argument("--session-lambda", type=float, default=300, 
                        help="Lambda parameter for Poisson distribution of inter-session times (default: 300)")
    parser.add_argument("--request-lambda", type=float, default=30, 
                        help="Lambda parameter for Poisson distribution of inter-request times (default: 30)")
    parser.add_argument("--session-scale", type=float, default=1.0,
                        help="Scaling factor for inter-session times (default: 1.0)")
    parser.add_argument("--request-scale", type=float, default=1.0,
                        help="Scaling factor for inter-request times (default: 1.0)")
    
    args = parser.parse_args()
    
    # Load input JSON
    with open(args.input_file, 'r') as f:
        data = json.load(f)
    
    # Generate timestamps
    timestamped_data = generate_timestamps(
        data, 
        args.session_lambda, 
        args.request_lambda, 
        args.session_scale, 
        args.request_scale
    )
    
    # Write output JSON
    with open(args.output_file, 'w') as f:
        json.dump(timestamped_data, f, indent=2)
    
    print(f"Added timestamps to {len(timestamped_data)} sessions")
    print(f"Session lambda: {args.session_lambda}, Request lambda: {args.request_lambda}")
    print(f"Session scale: {args.session_scale}, Request scale: {args.request_scale}")
    print(f"Output written to {args.output_file}")

if __name__ == "__main__":
    main()