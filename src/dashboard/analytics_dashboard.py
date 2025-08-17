"""Analytical dashboard for complaint insights."""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path


class AnalyticsDashboard:
    def __init__(self):
        self.df = None
        self._load_data()

    def _load_data(self):
        try:
            # Use only the working CSV file (parquet files are corrupted)
            data_paths = [
                Path("data/raw/complaints.csv")
            ]
            
            for data_path in data_paths:
                if data_path.exists():
                    try:
                        if data_path.suffix == '.csv':
                            self.df = pd.read_csv(data_path, encoding='utf-8', on_bad_lines='skip')
                        else:
                            self.df = pd.read_parquet(data_path)
                        st.success(f"Loaded data from {data_path}")
                        break
                    except Exception as e:
                        st.warning(f"Failed to load {data_path}: {e}")
                        continue
            else:
                st.error("Could not find any valid data file")
                self.df = pd.DataFrame()
                return

            if "Date received" in self.df.columns:
                self.df["Date received"] = pd.to_datetime(self.df["Date received"])

            st.success(f"Loaded {len(self.df)} complaint records")
        except Exception as e:
            st.error(f"Error loading data: {e}")
            self.df = pd.DataFrame()

    def render_overview_metrics(self):
        if self.df.empty:
            return

        st.subheader("📊 Key Metrics")
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Complaints", f"{len(self.df):,}")
        with col2:
            st.metric("Products", self.df["Product"].nunique())
        with col3:
            if "Date received" in self.df.columns:
                days = (
                    self.df["Date received"].max() - self.df["Date received"].min()
                ).days
                st.metric("Date Range (Days)", f"{days:,}")
        with col4:
            if "State" in self.df.columns:
                st.metric("States", self.df["State"].nunique())

    def render_product_analysis(self):
        if self.df.empty:
            return

        st.subheader("🏦 Product Analysis")
        product_counts = self.df["Product"].value_counts()

        col1, col2 = st.columns(2)
        with col1:
            fig = px.bar(
                x=product_counts.values,
                y=product_counts.index,
                orientation="h",
                title="Complaints by Product",
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig = px.pie(
                values=product_counts.values,
                names=product_counts.index,
                title="Product Distribution",
            )
            st.plotly_chart(fig, use_container_width=True)

    def run(self):
        st.title("📊 CrediTrust Complaint Analytics Dashboard")

        self.render_overview_metrics()
        st.markdown("---")
        self.render_product_analysis()


def main():
    st.set_page_config(page_title="CrediTrust Analytics", page_icon="📊", layout="wide")
    dashboard = AnalyticsDashboard()
    dashboard.run()


if __name__ == "__main__":
    main()
