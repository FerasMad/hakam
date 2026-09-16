# Hakam demo acceptance checklist

Run this short pass on the presentation machine after installing the backend
and frontend dependencies.

1. Open `http://localhost:8000/api/ready` and confirm `status` is `ready`, the
   four tasks are present, and the supplied thresholds are shown.
2. Upload one supported video with the incident near its midpoint and confirm a
   normal result shows vision output, real law evidence, and the Arabic ruling.
3. Exercise a known low-confidence case and confirm Human Review appears with
   no Law/RAG section or final ruling.
4. Switch between English and Arabic before and after analysis; confirm the
   result remains in place and the layout direction changes correctly.
5. Expand Technical Details, inspect the real contract and article IDs, and use
   Copy JSON to paste valid JSON into a text editor.
6. From an error or result, verify retry, replace video, and new review actions.
7. Check one desktop and one mobile viewport for readable video, evidence,
   confidence, Arabic text, and controls without horizontal page overflow.
8. Finish with one representative real football incident on the presentation
   hardware and note its end-to-end processing time.
