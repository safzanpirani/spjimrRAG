import { VALID_QUERY_CATEGORIES } from "../types/rag";

// Normalization helper
const normalize = (s: string) => (s || "").toLowerCase().trim();

// Central list of known PGPM terms and initiatives
const KNOWN_TERMS = [
  "abhyudaya", "docc", "sitaras", "samavesh", "intex",
  "aicte", "aacsb", "amba", "ppt", "cis",
  "mccombs", "insead", "cornell", "michigan",
  "barcelona", "reutlingen", "esic", "esb"
];

export const isKnownPGPMTerm = (query: string): boolean => {
  return KNOWN_TERMS.includes(normalize(query));
};

export const hasRelevantKeywords = (query: string): boolean => {
  const relevantKeywords = [
    "spjimr", "pgpm", "post graduate programme", "management",
    "admission", "eligibility", "curriculum", "fees", "placement",
    "duration", "campus", "accreditation", "ranking", "faculty"
  ];
  const q = normalize(query);
  return relevantKeywords.some(k => q.includes(k));
};

export const isFollowUpQuery = (query: string): boolean => {
  const patterns = [
    /^(longer|more|elaborate|tell me more|expand|details?)$/i,
    /^(what else|any other|additional|anything else)$/i,
    /^(it|this|that|the program|the course)$/i,
    /^(give me|show me|tell me) (more|all|everything|details)$/i
  ];
  return patterns.some(p => p.test(normalize(query)));
};

export const isStatsQuery = (query: string): boolean => {
  const patterns = [
    /\b(stats?|statistics|data|numbers|figures|all data)\b/i,
    /\b(give me all|show me all|tell me all)\b/i,
    /\b(complete (stats?|data|information))\b/i
  ];
  return patterns.some(p => p.test(query));
};

export const isBroadInformationQuery = (query: string): boolean => {
  const patterns = [
    /^(everything|tell me everything|all information)$/i,
    /^(complete|comprehensive|full) (info|information|details?)$/i,
    /^(overview|summary)$/i,
    /\b(tell me about|what is|what's) (pgpm|the program|this program)\b/i
  ];
  return patterns.some(p => p.test(normalize(query)));
};

export const suggestCategories = (query: string): string[] => {
  const q = normalize(query);
  const categories: string[] = [];
  const categoryKeywords: Record<string, string[]> = {
    eligibility: ["eligibility", "criteria", "requirement", "qualify", "experience", "marks", "degree"],
    admissions: ["admission", "apply", "application", "deadline", "process", "selection"],
    curriculum: ["curriculum", "course", "subject", "major", "minor", "academic", "syllabus"],
    fees: ["fees", "cost", "payment", "financial", "scholarship", "loan"],
    placements: ["placement", "job", "salary", "company", "career", "recruitment"],
    duration: ["duration", "length", "timeline", "phase", "semester"],
    campus: ["campus", "location", "facility", "infrastructure", "hostel"],
    accreditation: ["accreditation", "aacsb", "amba", "equis", "certification"],
    rankings: ["ranking", "rank", "rating", "reputation", "financial times"],
    faculty: ["faculty", "professor", "teacher", "staff", "instructor"]
  };
  for (const [category, keywords] of Object.entries(categoryKeywords)) {
    if (keywords.some(k => q.includes(k))) categories.push(category);
  }
  return categories;
};

// Enhanced Query Enhancement System (centralized)
export const enhanceQuery = async (originalQuery: string): Promise<string> => {
  const queryLower = normalize(originalQuery);
  const queryWords = queryLower.split(/\s+/).filter(word => word.length > 2);

  const baseContext = "SPJIMR PGPM program management";

  const needsEnhancement =
    originalQuery.length < 20 ||
    queryWords.length < 3 ||
    (!queryLower.includes('pgpm') && !queryLower.includes('spjimr')) ||
    isFollowUpQuery(originalQuery) ||
    isStatsQuery(originalQuery) ||
    isBroadInformationQuery(originalQuery);

  if (!needsEnhancement) return originalQuery;

  const domainExpansions: Record<string, string> = {
    curriculum: "curriculum courses subjects modules syllabus academic program structure",
    course: "courses curriculum subjects academic program modules",
    duration: "duration length time period months years program",
    admission: "admission admissions eligibility requirements application process criteria",
    eligibility: "eligibility criteria requirements qualification admission",
    deadline: "deadline dates timeline admission application last date",
    application: "application form process admission requirements procedure",
    fees: "fees cost tuition payment structure scholarship financial aid",
    cost: "cost fees expenses tuition financial charges",
    scholarship: "scholarship financial aid funding assistance fees support",
    payment: "payment fees cost installment structure financial",
    placement: "placement job career opportunities salary companies recruitment",
    salary: "salary placement packages compensation career opportunities",
    companies: "companies placement recruiters employers opportunities",
    career: "career placement opportunities job salary growth",
    campus: "campus facilities infrastructure location accommodation hostel",
    hostel: "hostel accommodation campus facilities residence",
    facilities: "facilities infrastructure campus amenities services",
    faculty: "faculty professors teachers teaching staff academics",
    teaching: "teaching faculty professors pedagogy learning methodology"
  };

  let expansions: string[] = [baseContext];
  for (const [domain, expansion] of Object.entries(domainExpansions)) {
    if (queryLower.includes(domain) || queryWords.some(w => w.includes(domain.substring(0, 4)))) {
      expansions.push(expansion);
    }
  }

  if (isStatsQuery(originalQuery)) {
    return `PGPM program statistics data numbers figures placement salary fees admission criteria duration curriculum comprehensive information ${originalQuery}`;
  }
  if (isBroadInformationQuery(originalQuery)) {
    return `PGPM program complete information overview summary comprehensive details curriculum admission placement fees faculty campus facilities ${originalQuery}`;
  }
  if (isFollowUpQuery(originalQuery)) {
    return `PGPM program detailed information comprehensive overview curriculum admission placement fees salary statistics faculty campus duration ${originalQuery}`;
  }

  if (queryWords.length <= 2) {
    const shortQueryExpansions: Record<string, string> = {
      'fees': 'PGPM program fees cost tuition structure payment financial',
      'admission': 'PGPM admission eligibility requirements application process criteria',
      'curriculum': 'PGPM curriculum courses subjects academic program structure',
      'placement': 'PGPM placement job career opportunities salary companies',
      'duration': 'PGPM program duration length time period months',
      'faculty': 'PGPM faculty professors teachers teaching staff',
      'campus': 'PGPM campus facilities infrastructure location',
      'eligibility': 'PGPM eligibility criteria requirements qualification admission',
      'salary': 'PGPM placement salary packages compensation career',
      'deadline': 'PGPM admission deadline dates timeline application',
      'scholarship': 'PGPM scholarship financial aid funding assistance fees',
      'stats': 'PGPM statistics data placement salary admission numbers figures',
      'everything': 'PGPM comprehensive information overview curriculum admission placement fees',
      'abhyudaya': 'PGPM Abhyudaya social projects community underprivileged local engagement development citizenship DoCC',
      'docc': 'PGPM DoCC Development Corporate Citizenship social projects organizations India community work',
      'sitaras': 'PGPM social projects community education learning experience Sitaras participants engagement'
    };
    for (const [key, expansion] of Object.entries(shortQueryExpansions)) {
      if (queryLower.includes(key)) {
        return `${expansion} ${originalQuery}`;
      }
    }
  }

  const enhanced = `${expansions.join(' ')} ${originalQuery}`;
  if (enhanced.length > 200) return `${baseContext} ${originalQuery}`;
  return enhanced;
};

export { KNOWN_TERMS };


