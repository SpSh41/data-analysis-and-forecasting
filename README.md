# Dataset Combining Script

This repository contains a practical example of how to combine two time-series datasets that come from different sources and do not have exactly the same structure.

The main goal of the script is to make the datasets consistent before stacking them together. It standardizes the date/time fields, aligns the shared columns, keeps track of where each row came from, and checks that no rows are accidentally lost during the process.

## What this script does

The script walks through the main steps needed to safely combine two datasets.

## Why this is useful

When working with long-term environmental or time-series data, datasets often come from different sources, have slightly different column names, or cover different time periods. If they are combined too quickly, it is easy to lose rows, duplicate timestamps, or overwrite useful information.

This script is meant to show a careful and transparent way to combine datasets while keeping track of what happened at each step.


