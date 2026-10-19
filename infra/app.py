"""Synthesize locally without querying an account or discovering credentials."""

from aws_cdk import App

from infra.foundation import FoundationStack
from infra.tracer import TracerStack

app = App(analytics_reporting=False)
FoundationStack(app)
TracerStack(app)
app.synth()
