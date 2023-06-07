const ModelEvaluation = artifacts.require("ModelEvaluation");
const ModelResultNFT = artifacts.require("ModelResultNFT");

module.exports = function(deployer) {
  deployer.deploy(ModelEvaluation);
  deployer.deploy(ModelResultNFT);
};
