// "Test yourself" incidents: the five highest-scoring test-set incidents from the full test
// (scripts/full_test.py, last-clip mode). In each, the model agreed with the referee on
// offence, card, action family and body part; ranked by the model's weakest confidence.
// Clips live in public/samples (committed; SoccerNet-MVFoul test split, used with KAUST's permission).

export type SampleIncident = {
  id: string;
  src: string;
  refereeKey: "refereeFoulCard";
  detailKey: "refereeTackleLowerBody";
};

export const sampleIncidents: SampleIncident[] = [
  { id: "88", src: "/samples/incident-88.mp4", refereeKey: "refereeFoulCard", detailKey: "refereeTackleLowerBody" },
  { id: "183", src: "/samples/incident-183.mp4", refereeKey: "refereeFoulCard", detailKey: "refereeTackleLowerBody" },
  { id: "144", src: "/samples/incident-144.mp4", refereeKey: "refereeFoulCard", detailKey: "refereeTackleLowerBody" },
  { id: "212", src: "/samples/incident-212.mp4", refereeKey: "refereeFoulCard", detailKey: "refereeTackleLowerBody" },
  { id: "281", src: "/samples/incident-281.mp4", refereeKey: "refereeFoulCard", detailKey: "refereeTackleLowerBody" },
];
