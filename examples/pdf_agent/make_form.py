"""Create the sample fillable order form used by the pdf_agent example.

Run this once before ``main.py`` (or let ``main.py`` do it for you):

    python examples/pdf_agent/make_form.py

Writes ``form.pdf`` next to this file with four fields:
``customer_name`` (text), ``quantity`` (text), ``priority`` (checkbox)
and ``size`` (radio group: Small / Large).
"""

from pathlib import Path

from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

OUT = Path(__file__).resolve().parent / "form.pdf"


def main() -> None:
    c = canvas.Canvas(str(OUT), pagesize=letter)
    c.setTitle("Sample Order Form")

    c.setFont("Helvetica-Bold", 16)
    c.drawString(72, 720, "Order Form")

    c.setFont("Helvetica", 11)
    c.drawString(72, 690, "Customer name:")
    c.acroForm.textfield(name="customer_name", x=200, y=680, width=250, height=20)

    c.drawString(72, 650, "Quantity:")
    c.acroForm.textfield(name="quantity", x=200, y=640, width=60, height=20)

    c.drawString(72, 610, "Priority shipping:")
    c.acroForm.checkbox(name="priority", x=200, y=600, buttonStyle="check")

    c.acroForm.radio(name="size", value="S", x=72, y=560, buttonStyle="circle")
    c.drawString(95, 562, "Small")
    c.acroForm.radio(name="size", value="L", x=160, y=560, buttonStyle="circle")
    c.drawString(183, 562, "Large")

    c.save()
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
