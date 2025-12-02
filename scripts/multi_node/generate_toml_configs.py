#!/usr/bin/env python3
import os
import sys
import argparse
from pathlib import Path

def read_template(template_path):
    """Read the template TOML file"""
    with open(template_path, 'r') as f:
        return f.read()

def generate_config(template_content, scale_factor):
    """Generate a new config with the specified scale factor"""
    # Replace the scale_factor value in the [app.client_toC] section
    lines = template_content.split('\n')
    new_lines = []
    
    for line in lines:
        if line.strip().startswith('config.scale_factor ='):
            new_lines.append(f'config.scale_factor = {scale_factor}')
        else:
            new_lines.append(line)
        # if line.strip() == '[app.client_toC]' or line.strip() == '[app.client_thinking]' or line.strip() == '[app.client_toB]' or line.strip() == '[app.client_coder]':
        #     in_client_section = True
        #     new_lines.append(line)
        # elif in_client_section and line.strip().startswith('config.scale_factor ='):
        #     new_lines.append(f'config.scale_factor = {scale_factor}')
        #     in_client_section = False  # Reset after modifying scale_factor
        # else:
        #     new_lines.append(line)
    
    return '\n'.join(new_lines)

def main():
    parser = argparse.ArgumentParser(description='Generate TOML configs with different scale factors')
    parser.add_argument('--template', required=True, help='Path to template TOML file')
    parser.add_argument('--output-dir', required=True, help='Output directory for generated configs')
    parser.add_argument('--scale-factors', required=True, nargs='+', type=float, 
                       help='List of scale factors to generate')
    
    args = parser.parse_args()
    
    # Read template
    template_content = read_template(args.template)
    
    # Create output directory if it doesn't exist
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Generate configs for each scale factor
    for scale_factor in args.scale_factors:
        config_content = generate_config(template_content, scale_factor)
        
        # Create filename based on scale factor
        filename = f"bailian_clients_sf{scale_factor}.toml"
        output_path = os.path.join(args.output_dir, filename)
        
        # Write the generated config
        with open(output_path, 'w') as f:
            f.write(config_content)
        
        print(f"Generated: {output_path}")

if __name__ == "__main__":
    main()
