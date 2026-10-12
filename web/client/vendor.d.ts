// mammoth ships no types; only extractRawText is used (client/parse.ts).
declare module "mammoth/mammoth.browser.js" {
  const mammoth: { extractRawText(input: { arrayBuffer: ArrayBuffer }): Promise<{ value: string }> };
  export default mammoth;
}
